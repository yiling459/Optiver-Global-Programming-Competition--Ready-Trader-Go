# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
import asyncio
import itertools
import pickle
import numpy as np
from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side

import pandas as pd
import csv
import numpy as np
LOT_SIZE = 10
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS


class AutoTrader(BaseAutoTrader):
    """Example Auto-trader.

    When it starts this auto-trader places ten-lot bid and ask orders at the
    current best-bid and best-ask prices respectively. Thereafter, if it has
    a long position (it has bought more lots than it has sold) it reduces its
    bid and ask prices. Conversely, if it has a short position (it has sold
    more lots than it has bought) then it increases its bid and ask prices.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0
        
        self.big_bill_ask_FUTURE=0
        self.big_bill_bid_FUTURE=0
        self.big_bill_ask_ETF=0
        self.big_bill_bid_ETF=0
        self.history=[0,0,0,0,0]
        #with open('output.csv',mode='w',newline='') as file:
            #writer=csv.writer(file)
            #writer.writerow(['ask_prices','ask_volumes','bid_prices','bid_volumes'])

        self.para = np.array([4.99325349e-01,4.99325349e-01,-1.63716662e-01,1.63717294e-01,
        7.11188053e-04,-3.24373155e-04,-3.56691402e-02, -3.34521759e-02,
         -2.47186476e-02, -2.70819467e-02, -2.21935806e-02, 1.71802162e-02])
        self.para2=np.array([4.98896958e-01, 4.98896958e-01, -3.28393549e-01, 3.27647082e-01,
        1.67055391e-03,-1.94629469e-04, -3.05817559e-02, -3.46504459e-02,
        -2.19845297e-02, -2.52628973e-02, -2.01023961e-02,1.58686716e-02])
        self.para3=np.array([
            4.98612058e-01,  4.98612058e-01, -4.93371940e-01,  4.90853370e-01,
  3.64612492e-03, -3.14051903e-04, -2.51046112e-02, -3.57968672e-02,
 -1.92988411e-02, -2.29557092e-02, -1.80078616e-02,  1.44601284e-02,
        ])
    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.bids or client_order_id in self.asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your hedge orders is filled.

        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received hedge filled for order %d with average price %d and volume %d", client_order_id,
                         price, volume)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        self.logger.info("received order book for instrument %d with sequence number %d", instrument,
                         sequence_number)
        #the variables that determine the big bills 
        if instrument==Instrument.ETF and abs(max(ask_volumes))>8000:
            self.big_bill_ask_ETF=1
        if instrument==Instrument.ETF and abs(max(bid_volumes))>8000:
            self.big_bill_bid_ETF=1
        if instrument==Instrument.FUTURE and abs(max(ask_volumes))>18000:
            self.big_bill_ask_FUTURE=1
        if instrument==Instrument.FUTURE and abs(max(bid_volumes))>18000:
            self.big_bill_bid_FUTURE=1
        #the difference between the ask volume and bid volume
        difference=sum(ask_volumes)-sum(bid_volumes)
        
        row_total=[instrument,sequence_number,ask_prices,ask_volumes,bid_prices,bid_volumes,self.big_bill_bid_FUTURE,self.big_bill_ask_FUTURE,self.big_bill_bid_ETF,self.big_bill_ask_ETF,difference]


        #make prediction to the next market price change, next next market price change and next nexst next market price change using the trained model
        if instrument==Instrument.ETF:
            print('ETF')
            '''input_lst=[ask_prices[0],ask_prices[1],ask_prices[2],
                       ask_prices[3],ask_prices[4],ask_volumes[1],
                       ask_volumes[2],ask_volumes[3],ask_volumes[4],ask_volumes[5],difference]
            array_input=np.array(input_lst)'''
            current_market_value=(ask_prices[0]+bid_prices[0])/2
            difference=sum(ask_volumes)-sum(bid_volumes)
        # print(ask_prices[0],ask_prices[1],ask_prices[2],
        #                ask_prices[3],ask_prices[4],ask_volumes[1],
        #                ask_volumes[2],ask_volumes[3],ask_volumes[4],ask_volumes[5],difference)
            input_lst=[ask_prices[0],ask_prices[0],ask_prices[1],ask_prices[2],
                       ask_prices[3],ask_prices[4],ask_volumes[0],
                       ask_volumes[1],ask_volumes[2],ask_volumes[3],ask_volumes[4],difference]
            print(current_market_value)
            print(input_lst)
            print(self.para)
            array_input=np.array(input_lst)
            pred = np.dot(array_input,self.para) + 149.21758120827144
            print("333")
            self.logger.info("prediction:" + str(pred))
            print(pred)
            pred2=np.dot(array_input,self.para2)+171.80880311250803
            self.logger.info('prediction 2:'+str(pred2))
            print(pred2)
            pred3=np.dot(array_input,self.para3)+187.99534604750806
            #self.logger.info('prediction 3:'+str(pred3))
            print(pred3)
            if pred>current_market_value and (pred2<pred or pred3<pred) and self.position>-POSITION_LIMIT:
                self.ask_id=next(self.order_ids)
                print('1')
                price_adjustment = 2
                print('2')
                new_ask_price = ask_prices[0] + price_adjustment if ask_prices[0] != 0 else 0
                print('3')
                self.send_insert_order(self.ask_id,Side.SELL,new_ask_price,LOT_SIZE,Lifespan.FILL_AND_KILL)
                print('4')
                self.asks.add(self.ask_id)
                print('5')
            if pred3>pred2 and pred2>pred and pred>current_market_value and self.position>-POSITION_LIMIT:
                self.ask_id=next(self.order_ids)
                print('1')
                price_adjustment=5
                print('2')
                new_ask_price=ask_prices[0]+price_adjustment
                print('3')
                self.send_insert_order(self.ask_id,Side.SELL,new_ask_price,LOT_SIZE,Lifespan.GOOD_FOR_DAY)
                print('4')
                self.asks.add(self.ask_id)
                print('5')
            if pred<current_market_value and (pred2>pred or pred3>pred) and self.position<POSITION_LIMIT:
                self.bid_id = next(self.order_ids)
                price_adjustment=2
                new_bid_price=bid_prices[0]-price_adjustment
                self.send_insert_order(self.bid_id,Side.BUY,new_bid_price,LOT_SIZE,Lifespan.FILL_AND_KILL)
                self.bids.add(self.bid_id)
                print('done')
            if pred3<pred2 and pred2<pred and pred<current_market_value and self.position<POSITION_LIMIT:
                self.bid_id = next(self.order_ids)
                price_adjustment=5
                new_bid_price=bid_prices[0]-price_adjustment
                self.send_insert_order(self.bid_id,Side.BUY,new_bid_price,LOT_SIZE,Lifespan.GOOD_FOR_DAY)
                self.bids.add(self.bid_id)
                print('done')
            print('done')
        #write the data into the csv file
        
        '''with open('output4.csv',mode='a',newline='') as file:
            writer=csv.writer(file)
            writer.writerow(row_total)
            
        if instrument==Instrument.ETF:
            row_ETF=[instrument,sequence_number,ask_prices,ask_volumes,bid_prices,bid_volumes,self.big_bill_ask_ETF,self.big_bill_bid_ETF,difference]
            with open('outputETF4.csv',mode='a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow(row_ETF)
        if instrument==Instrument.FUTURE:
            row_FUTURE=[instrument,sequence_number,ask_prices,ask_volumes,bid_prices,bid_volumes,self.big_bill_ask_FUTURE,self.big_bill_bid_FUTURE,difference]
            with open('outputFuture4.csv',mode='a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow(row_FUTURE)'''
        
        if instrument == Instrument.FUTURE:
            
            print('future')
            price_adjustment = (self.position // LOT_SIZE) * TICK_SIZE_IN_CENTS
            #here we change + to - in order to buy at a lower price
            new_bid_price = bid_prices[0] - price_adjustment if bid_prices[0] != 0 else 0
            new_ask_price = ask_prices[0] + price_adjustment if ask_prices[0] != 0 else 0

            #the code added by xjh
            ma=(bid_prices[0]+ask_prices[0])/2
            self.history.pop(0)
            self.history.append(ma)
            volatile=np.std(np.array(self.history))
            stop_loss=ma-2*volatile
            take_profit=ma+2*volatile


            if self.bid_id != 0 and new_bid_price not in (self.bid_price, 0):
                self.send_cancel_order(self.bid_id)
                self.bid_id = 0
            if self.ask_id != 0 and new_ask_price not in (self.ask_price, 0):
                self.send_cancel_order(self.ask_id)
                self.ask_id = 0

            if self.bid_id == 0 and new_bid_price != 0 and self.position < POSITION_LIMIT:
                self.bid_id = next(self.order_ids)
                self.bid_price = new_bid_price
                self.send_insert_order(self.bid_id, Side.BUY, new_bid_price, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.bids.add(self.bid_id)

            if self.ask_id == 0 and new_ask_price != 0 and self.position > -POSITION_LIMIT:
                self.ask_id = next(self.order_ids)
                self.ask_price = new_ask_price
                self.send_insert_order(self.ask_id, Side.SELL, new_ask_price, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.asks.add(self.ask_id)

            #call the volatile
            if self.bid_id<=1 and bid_prices[0]<=stop_loss and self.position<POSITION_LIMIT:
                self.bid_id=next(self.order_ids)
                self.bid_price=bid_prices[0]
                self.send_insert_order(self.bid_id,Side.BUY,self.bid_price,LOT_SIZE,Lifespan.GOOD_FOR_DAY)
                self.bids.add(self.bid_id)

            if self.ask_id<=1 and ask_prices[0]>=stop_loss and self.position>-POSITION_LIMIT:
                self.ask_id=next(self.order_ids)
                self.ask_price=ask_prices[0]
                self.send_insert_order(self.ask_id,Side.SELL,self.ask_price,LOT_SIZE,Lifespan.GOOD_FOR_DAY)
                self.asks.add(self.ask_id)

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        if client_order_id in self.bids:
            self.position += volume
            self.send_hedge_order(next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, volume)
        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(next(self.order_ids), Side.BID, MAX_ASK_NEAREST_TICK, volume)

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                         client_order_id, fill_volume, remaining_volume, fees)
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0

            # It could be either a bid or an ask
            self.bids.discard(client_order_id)
            self.asks.discard(client_order_id)

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        self.logger.info("received trade ticks for instrument %d with sequence number %d", instrument,
                         sequence_number)
