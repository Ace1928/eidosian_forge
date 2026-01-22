from __future__ import annotations
import copy
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import NDFrameT
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
import pandas.core.algorithms as algos
from pandas.core.apply import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.generic import (
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import (
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.core.indexes.api import MultiIndex
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.period import (
from pandas.core.indexes.timedeltas import (
from pandas.tseries.frequencies import (
from pandas.tseries.offsets import (
def _adjust_dates_anchored(first: Timestamp, last: Timestamp, freq: Tick, closed: Literal['right', 'left']='right', origin: TimeGrouperOrigin='start_day', offset: Timedelta | None=None, unit: str='ns') -> tuple[Timestamp, Timestamp]:
    first = first.as_unit(unit)
    last = last.as_unit(unit)
    if offset is not None:
        offset = offset.as_unit(unit)
    freq_value = Timedelta(freq).as_unit(unit)._value
    origin_timestamp = 0
    if origin == 'start_day':
        origin_timestamp = first.normalize()._value
    elif origin == 'start':
        origin_timestamp = first._value
    elif isinstance(origin, Timestamp):
        origin_timestamp = origin.as_unit(unit)._value
    elif origin in ['end', 'end_day']:
        origin_last = last if origin == 'end' else last.ceil('D')
        sub_freq_times = (origin_last._value - first._value) // freq_value
        if closed == 'left':
            sub_freq_times += 1
        first = origin_last - sub_freq_times * freq
        origin_timestamp = first._value
    origin_timestamp += offset._value if offset else 0
    first_tzinfo = first.tzinfo
    last_tzinfo = last.tzinfo
    if first_tzinfo is not None:
        first = first.tz_convert('UTC')
    if last_tzinfo is not None:
        last = last.tz_convert('UTC')
    foffset = (first._value - origin_timestamp) % freq_value
    loffset = (last._value - origin_timestamp) % freq_value
    if closed == 'right':
        if foffset > 0:
            fresult_int = first._value - foffset
        else:
            fresult_int = first._value - freq_value
        if loffset > 0:
            lresult_int = last._value + (freq_value - loffset)
        else:
            lresult_int = last._value
    else:
        if foffset > 0:
            fresult_int = first._value - foffset
        else:
            fresult_int = first._value
        if loffset > 0:
            lresult_int = last._value + (freq_value - loffset)
        else:
            lresult_int = last._value + freq_value
    fresult = Timestamp(fresult_int, unit=unit)
    lresult = Timestamp(lresult_int, unit=unit)
    if first_tzinfo is not None:
        fresult = fresult.tz_localize('UTC').tz_convert(first_tzinfo)
    if last_tzinfo is not None:
        lresult = lresult.tz_localize('UTC').tz_convert(last_tzinfo)
    return (fresult, lresult)