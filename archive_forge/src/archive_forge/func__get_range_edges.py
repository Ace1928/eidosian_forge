from __future__ import annotations
import datetime
import typing
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.types import SideOptions
def _get_range_edges(first: CFTimeDatetime, last: CFTimeDatetime, freq: BaseCFTimeOffset, closed: SideOptions='left', origin: str | CFTimeDatetime='start_day', offset: datetime.timedelta | None=None):
    """Get the correct starting and ending datetimes for the resampled
    CFTimeIndex range.

    Parameters
    ----------
    first : cftime.datetime
        Uncorrected starting datetime object for resampled CFTimeIndex range.
        Usually the min of the original CFTimeIndex.
    last : cftime.datetime
        Uncorrected ending datetime object for resampled CFTimeIndex range.
        Usually the max of the original CFTimeIndex.
    freq : xarray.coding.cftime_offsets.BaseCFTimeOffset
        The offset object representing target conversion a.k.a. resampling
        frequency. Contains information on offset type (e.g. Day or 'D') and
        offset magnitude (e.g., n = 3).
    closed : 'left' or 'right'
        Which side of bin interval is closed. Defaults to 'left'.
    origin : {'epoch', 'start', 'start_day', 'end', 'end_day'} or cftime.datetime, default 'start_day'
        The datetime on which to adjust the grouping. The timezone of origin
        must match the timezone of the index.

        If a datetime is not used, these values are also supported:
        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
        - 'end': `origin` is the last value of the timeseries
        - 'end_day': `origin` is the ceiling midnight of the last day
    offset : datetime.timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    first : cftime.datetime
        Corrected starting datetime object for resampled CFTimeIndex range.
    last : cftime.datetime
        Corrected ending datetime object for resampled CFTimeIndex range.
    """
    if isinstance(freq, Tick):
        first, last = _adjust_dates_anchored(first, last, freq, closed=closed, origin=origin, offset=offset)
        return (first, last)
    else:
        first = normalize_date(first)
        last = normalize_date(last)
    first = freq.rollback(first) if closed == 'left' else first - freq
    last = last + freq
    return (first, last)