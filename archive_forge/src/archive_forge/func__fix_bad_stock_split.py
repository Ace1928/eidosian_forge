import datetime as _datetime
import dateutil as _dateutil
import logging
import numpy as np
import pandas as pd
import time as _time
from yfinance import shared, utils
from yfinance.const import _BASE_URL_, _PRICE_COLNAMES_
@utils.log_indent_decorator
def _fix_bad_stock_split(self, df, interval, tz_exchange):
    if df.empty:
        return df
    logger = utils.get_yf_logger()
    interday = interval in ['1d', '1wk', '1mo', '3mo']
    if not interday:
        return df
    df = df.sort_index(ascending=False)
    split_f = df['Stock Splits'].to_numpy() != 0
    if not split_f.any():
        logger.debug('price-repair-split: No splits in data')
        return df
    most_recent_split_day = df.index[split_f].max()
    split = df.loc[most_recent_split_day, 'Stock Splits']
    if most_recent_split_day == df.index[0]:
        logger.info("price-repair-split: Need 1+ day of price data after split to determine true price. Won't repair")
        return df
    logger.debug(f'price-repair-split: Most recent split = {split:.4f} @ {most_recent_split_day.date()}')
    return self._fix_prices_sudden_change(df, interval, tz_exchange, split, correct_volume=True)