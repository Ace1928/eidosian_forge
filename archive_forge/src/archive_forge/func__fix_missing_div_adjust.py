import datetime as _datetime
import dateutil as _dateutil
import logging
import numpy as np
import pandas as pd
import time as _time
from yfinance import shared, utils
from yfinance.const import _BASE_URL_, _PRICE_COLNAMES_
@utils.log_indent_decorator
def _fix_missing_div_adjust(self, df, interval, tz_exchange):
    if df.empty:
        return df
    logger = utils.get_yf_logger()
    if df is None or df.empty:
        return df
    interday = interval in ['1d', '1wk', '1mo', '3mo']
    if not interday:
        return df
    df = df.sort_index()
    f_div = (df['Dividends'] != 0.0).to_numpy()
    if not f_div.any():
        logger.debug('div-adjust-repair: No dividends to check')
        return df
    df2 = df.copy()
    if df2.index.tz is None:
        df2.index = df2.index.tz_localize(tz_exchange)
    elif df2.index.tz != tz_exchange:
        df2.index = df2.index.tz_convert(tz_exchange)
    div_indices = np.where(f_div)[0]
    last_div_idx = div_indices[-1]
    if last_div_idx == 0:
        logger.debug('div-adjust-repair: Insufficient data to recalculate div-adjustment')
        return df
    if len(div_indices) == 1:
        prev_idx = 0
        prev_dt = None
    else:
        prev_idx = div_indices[-2]
        prev_dt = df2.index[prev_idx]
    f_no_adj = (df2['Close'] == df2['Adj Close']).to_numpy()[prev_idx:last_div_idx]
    threshold_pct = 0.5
    Yahoo_failed = np.sum(f_no_adj) / len(f_no_adj) > threshold_pct
    if Yahoo_failed:
        last_div_dt = df2.index[last_div_idx]
        last_div_row = df2.loc[last_div_dt]
        close_day_before = df2['Close'].iloc[last_div_idx - 1]
        adj = 1.0 - df2['Dividends'].iloc[last_div_idx] / close_day_before
        div = last_div_row['Dividends']
        msg = f'Correcting missing div-adjustment preceding div = {div} @ {last_div_dt.date()} (prev_dt={prev_dt})'
        logger.debug('div-adjust-repair: ' + msg)
        if interval == '1d':
            df2.loc[:last_div_dt - _datetime.timedelta(seconds=1), 'Adj Close'] *= adj
        else:
            df2.loc[:last_div_dt, 'Adj Close'] *= adj
    return df2