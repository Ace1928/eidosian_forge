from __future__ import annotations
import re
import warnings
from collections.abc import Hashable
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Callable, Union
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime, OutOfBoundsTimedelta
from xarray.coding.variables import (
from xarray.core import indexing
from xarray.core.common import contains_cftime_datetimes, is_np_datetime_like
from xarray.core.duck_array_ops import asarray
from xarray.core.formatting import first_n_items, format_timestamp, last_item
from xarray.core.pdcompat import nanosecond_precision_timestamp
from xarray.core.utils import emit_user_level_warning
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import T_ChunkedArray, get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
from xarray.namedarray.utils import is_duck_dask_array
def _eagerly_encode_cf_datetime(dates: T_DuckArray, units: str | None=None, calendar: str | None=None, dtype: np.dtype | None=None, allow_units_modification: bool=True) -> tuple[T_DuckArray, str, str]:
    dates = asarray(dates)
    data_units = infer_datetime_units(dates)
    if units is None:
        units = data_units
    else:
        units = _cleanup_netcdf_time_units(units)
    if calendar is None:
        calendar = infer_calendar_name(dates)
    try:
        if not _is_standard_calendar(calendar) or dates.dtype.kind == 'O':
            raise OutOfBoundsDatetime
        assert dates.dtype == 'datetime64[ns]'
        time_units, ref_date = _unpack_time_units_and_ref_date(units)
        time_delta = _time_units_to_timedelta64(time_units)
        dates_as_index = pd.DatetimeIndex(dates.ravel())
        time_deltas = dates_as_index - ref_date
        needed_units, data_ref_date = _unpack_time_units_and_ref_date(data_units)
        if data_units != units:
            ref_delta = abs(data_ref_date - ref_date).to_timedelta64()
            data_delta = _time_units_to_timedelta64(needed_units)
            if ref_delta % data_delta > np.timedelta64(0, 'ns'):
                needed_units = _infer_time_units_from_diff(ref_delta)
        needed_time_delta = _time_units_to_timedelta64(needed_units)
        floor_division = True
        if time_delta > needed_time_delta:
            floor_division = False
            if dtype is None:
                emit_user_level_warning(f"Times can't be serialized faithfully to int64 with requested units {units!r}. Resolution of {needed_units!r} needed. Serializing times to floating point instead. Set encoding['dtype'] to integer dtype to serialize to int64. Set encoding['dtype'] to floating point dtype to silence this warning.")
            elif np.issubdtype(dtype, np.integer) and allow_units_modification:
                new_units = f'{needed_units} since {format_timestamp(ref_date)}'
                emit_user_level_warning(f"Times can't be serialized faithfully to int64 with requested units {units!r}. Serializing with units {new_units!r} instead. Set encoding['dtype'] to floating point dtype to serialize with units {units!r}. Set encoding['units'] to {new_units!r} to silence this warning .")
                units = new_units
                time_delta = needed_time_delta
                floor_division = True
        num = _division(time_deltas, time_delta, floor_division)
        num = num.values.reshape(dates.shape)
    except (OutOfBoundsDatetime, OverflowError, ValueError):
        num = _encode_datetime_with_cftime(dates, units, calendar)
        num = cast_to_int_if_safe(num)
    if dtype is not None:
        num = _cast_to_dtype_if_safe(num, dtype)
    return (num, units, calendar)