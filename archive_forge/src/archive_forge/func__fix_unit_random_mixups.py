import datetime as _datetime
import dateutil as _dateutil
import logging
import numpy as np
import pandas as pd
import time as _time
from yfinance import shared, utils
from yfinance.const import _BASE_URL_, _PRICE_COLNAMES_
@utils.log_indent_decorator
def _fix_unit_random_mixups(self, df, interval, tz_exchange, prepost):
    if df.empty:
        return df
    logger = utils.get_yf_logger()
    if df.shape[0] == 0:
        if 'Repaired?' not in df.columns:
            df['Repaired?'] = False
        return df
    if df.shape[0] == 1:
        logger.info('price-repair-100x: Cannot check single-row table for 100x price errors')
        if 'Repaired?' not in df.columns:
            df['Repaired?'] = False
        return df
    df2 = df.copy()
    if df2.index.tz is None:
        df2.index = df2.index.tz_localize(tz_exchange)
    elif df2.index.tz != tz_exchange:
        df2.index = df2.index.tz_convert(tz_exchange)
    from scipy import ndimage as _ndimage
    data_cols = ['High', 'Open', 'Low', 'Close', 'Adj Close']
    data_cols = [c for c in data_cols if c in df2.columns]
    f_zeroes = (df2[data_cols] == 0).any(axis=1).to_numpy()
    if f_zeroes.any():
        df2_zeroes = df2[f_zeroes]
        df2 = df2[~f_zeroes]
        df = df[~f_zeroes]
    else:
        df2_zeroes = None
    if df2.shape[0] <= 1:
        logger.info('price-repair-100x: Insufficient good data for detecting 100x price errors')
        if 'Repaired?' not in df.columns:
            df['Repaired?'] = False
        return df
    df2_data = df2[data_cols].to_numpy()
    median = _ndimage.median_filter(df2_data, size=(3, 3), mode='wrap')
    ratio = df2_data / median
    ratio_rounded = (ratio / 20).round() * 20
    f = ratio_rounded == 100
    ratio_rcp = 1.0 / ratio
    ratio_rcp_rounded = (ratio_rcp / 20).round() * 20
    f_rcp = (ratio_rounded == 100) | (ratio_rcp_rounded == 100)
    f_either = f | f_rcp
    if not f_either.any():
        logger.info('price-repair-100x: No sporadic 100x errors')
        if 'Repaired?' not in df.columns:
            df['Repaired?'] = False
        return df
    tag = -1.0
    for i in range(len(data_cols)):
        fi = f_either[:, i]
        c = data_cols[i]
        df2.loc[fi, c] = tag
    n_before = (df2_data == tag).sum()
    df2 = self._reconstruct_intervals_batch(df2, interval, prepost, tag)
    df2_tagged = df2[data_cols].to_numpy() == tag
    n_after = (df2[data_cols].to_numpy() == tag).sum()
    if n_after > 0:
        f = (df2[data_cols].to_numpy() == tag) & f
        for i in range(f.shape[0]):
            fi = f[i, :]
            if not fi.any():
                continue
            idx = df2.index[i]
            for c in ['Open', 'Close']:
                j = data_cols.index(c)
                if fi[j]:
                    df2.loc[idx, c] = df.loc[idx, c] * 0.01
            c = 'High'
            j = data_cols.index(c)
            if fi[j]:
                df2.loc[idx, c] = df2.loc[idx, ['Open', 'Close']].max()
            c = 'Low'
            j = data_cols.index(c)
            if fi[j]:
                df2.loc[idx, c] = df2.loc[idx, ['Open', 'Close']].min()
        f_rcp = (df2[data_cols].to_numpy() == tag) & f_rcp
        for i in range(f_rcp.shape[0]):
            fi = f_rcp[i, :]
            if not fi.any():
                continue
            idx = df2.index[i]
            for c in ['Open', 'Close']:
                j = data_cols.index(c)
                if fi[j]:
                    df2.loc[idx, c] = df.loc[idx, c] * 100.0
            c = 'High'
            j = data_cols.index(c)
            if fi[j]:
                df2.loc[idx, c] = df2.loc[idx, ['Open', 'Close']].max()
            c = 'Low'
            j = data_cols.index(c)
            if fi[j]:
                df2.loc[idx, c] = df2.loc[idx, ['Open', 'Close']].min()
        df2_tagged = df2[data_cols].to_numpy() == tag
        n_after_crude = df2_tagged.sum()
    else:
        n_after_crude = n_after
    n_fixed = n_before - n_after_crude
    n_fixed_crudely = n_after - n_after_crude
    if n_fixed > 0:
        report_msg = f'{self.ticker}: fixed {n_fixed}/{n_before} currency unit mixups '
        if n_fixed_crudely > 0:
            report_msg += f'({n_fixed_crudely} crudely) '
        report_msg += f'in {interval} price data'
        logger.info('price-repair-100x: ' + report_msg)
    f_either = df2[data_cols].to_numpy() == tag
    for j in range(len(data_cols)):
        fj = f_either[:, j]
        if fj.any():
            c = data_cols[j]
            df2.loc[fj, c] = df.loc[fj, c]
    if df2_zeroes is not None:
        if 'Repaired?' not in df2_zeroes.columns:
            df2_zeroes['Repaired?'] = False
        df2 = pd.concat([df2, df2_zeroes]).sort_index()
        df2.index = pd.to_datetime(df2.index)
    return df2