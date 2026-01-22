from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
def _stata_elapsed_date_to_datetime_vec(dates: Series, fmt: str) -> Series:
    """
    Convert from SIF to datetime. https://www.stata.com/help.cgi?datetime

    Parameters
    ----------
    dates : Series
        The Stata Internal Format date to convert to datetime according to fmt
    fmt : str
        The format to convert to. Can be, tc, td, tw, tm, tq, th, ty
        Returns

    Returns
    -------
    converted : Series
        The converted dates

    Examples
    --------
    >>> dates = pd.Series([52])
    >>> _stata_elapsed_date_to_datetime_vec(dates , "%tw")
    0   1961-01-01
    dtype: datetime64[ns]

    Notes
    -----
    datetime/c - tc
        milliseconds since 01jan1960 00:00:00.000, assuming 86,400 s/day
    datetime/C - tC - NOT IMPLEMENTED
        milliseconds since 01jan1960 00:00:00.000, adjusted for leap seconds
    date - td
        days since 01jan1960 (01jan1960 = 0)
    weekly date - tw
        weeks since 1960w1
        This assumes 52 weeks in a year, then adds 7 * remainder of the weeks.
        The datetime value is the start of the week in terms of days in the
        year, not ISO calendar weeks.
    monthly date - tm
        months since 1960m1
    quarterly date - tq
        quarters since 1960q1
    half-yearly date - th
        half-years since 1960h1 yearly
    date - ty
        years since 0000
    """
    MIN_YEAR, MAX_YEAR = (Timestamp.min.year, Timestamp.max.year)
    MAX_DAY_DELTA = (Timestamp.max - datetime(1960, 1, 1)).days
    MIN_DAY_DELTA = (Timestamp.min - datetime(1960, 1, 1)).days
    MIN_MS_DELTA = MIN_DAY_DELTA * 24 * 3600 * 1000
    MAX_MS_DELTA = MAX_DAY_DELTA * 24 * 3600 * 1000

    def convert_year_month_safe(year, month) -> Series:
        """
        Convert year and month to datetimes, using pandas vectorized versions
        when the date range falls within the range supported by pandas.
        Otherwise it falls back to a slower but more robust method
        using datetime.
        """
        if year.max() < MAX_YEAR and year.min() > MIN_YEAR:
            return to_datetime(100 * year + month, format='%Y%m')
        else:
            index = getattr(year, 'index', None)
            return Series([datetime(y, m, 1) for y, m in zip(year, month)], index=index)

    def convert_year_days_safe(year, days) -> Series:
        """
        Converts year (e.g. 1999) and days since the start of the year to a
        datetime or datetime64 Series
        """
        if year.max() < MAX_YEAR - 1 and year.min() > MIN_YEAR:
            return to_datetime(year, format='%Y') + to_timedelta(days, unit='d')
        else:
            index = getattr(year, 'index', None)
            value = [datetime(y, 1, 1) + timedelta(days=int(d)) for y, d in zip(year, days)]
            return Series(value, index=index)

    def convert_delta_safe(base, deltas, unit) -> Series:
        """
        Convert base dates and deltas to datetimes, using pandas vectorized
        versions if the deltas satisfy restrictions required to be expressed
        as dates in pandas.
        """
        index = getattr(deltas, 'index', None)
        if unit == 'd':
            if deltas.max() > MAX_DAY_DELTA or deltas.min() < MIN_DAY_DELTA:
                values = [base + timedelta(days=int(d)) for d in deltas]
                return Series(values, index=index)
        elif unit == 'ms':
            if deltas.max() > MAX_MS_DELTA or deltas.min() < MIN_MS_DELTA:
                values = [base + timedelta(microseconds=int(d) * 1000) for d in deltas]
                return Series(values, index=index)
        else:
            raise ValueError('format not understood')
        base = to_datetime(base)
        deltas = to_timedelta(deltas, unit=unit)
        return base + deltas
    bad_locs = np.isnan(dates)
    has_bad_values = False
    if bad_locs.any():
        has_bad_values = True
        dates._values[bad_locs] = 1.0
    dates = dates.astype(np.int64)
    if fmt.startswith(('%tc', 'tc')):
        base = stata_epoch
        ms = dates
        conv_dates = convert_delta_safe(base, ms, 'ms')
    elif fmt.startswith(('%tC', 'tC')):
        warnings.warn('Encountered %tC format. Leaving in Stata Internal Format.', stacklevel=find_stack_level())
        conv_dates = Series(dates, dtype=object)
        if has_bad_values:
            conv_dates[bad_locs] = NaT
        return conv_dates
    elif fmt.startswith(('%td', 'td', '%d', 'd')):
        base = stata_epoch
        days = dates
        conv_dates = convert_delta_safe(base, days, 'd')
    elif fmt.startswith(('%tw', 'tw')):
        year = stata_epoch.year + dates // 52
        days = dates % 52 * 7
        conv_dates = convert_year_days_safe(year, days)
    elif fmt.startswith(('%tm', 'tm')):
        year = stata_epoch.year + dates // 12
        month = dates % 12 + 1
        conv_dates = convert_year_month_safe(year, month)
    elif fmt.startswith(('%tq', 'tq')):
        year = stata_epoch.year + dates // 4
        quarter_month = dates % 4 * 3 + 1
        conv_dates = convert_year_month_safe(year, quarter_month)
    elif fmt.startswith(('%th', 'th')):
        year = stata_epoch.year + dates // 2
        month = dates % 2 * 6 + 1
        conv_dates = convert_year_month_safe(year, month)
    elif fmt.startswith(('%ty', 'ty')):
        year = dates
        first_month = np.ones_like(dates)
        conv_dates = convert_year_month_safe(year, first_month)
    else:
        raise ValueError(f'Date fmt {fmt} not understood')
    if has_bad_values:
        conv_dates[bad_locs] = NaT
    return conv_dates