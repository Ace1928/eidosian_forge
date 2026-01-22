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
def _decode_datetime_with_pandas(flat_num_dates: np.ndarray, units: str, calendar: str) -> np.ndarray:
    if not _is_standard_calendar(calendar):
        raise OutOfBoundsDatetime(f'Cannot decode times from a non-standard calendar, {calendar!r}, using pandas.')
    time_units, ref_date = _unpack_netcdf_time_units(units)
    time_units = _netcdf_to_numpy_timeunit(time_units)
    try:
        ref_date = nanosecond_precision_timestamp(ref_date)
    except ValueError:
        raise OutOfBoundsDatetime
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)
        if flat_num_dates.size > 0:
            pd.to_timedelta(flat_num_dates.min(), time_units) + ref_date
            pd.to_timedelta(flat_num_dates.max(), time_units) + ref_date
    if flat_num_dates.dtype.kind in 'iu':
        flat_num_dates = flat_num_dates.astype(np.int64)
    nan = np.isnan(flat_num_dates) | (flat_num_dates == np.iinfo(np.int64).min)
    flat_num_dates = flat_num_dates * _NS_PER_TIME_DELTA[time_units]
    flat_num_dates_ns_int = np.zeros_like(flat_num_dates, dtype=np.int64)
    flat_num_dates_ns_int[nan] = np.iinfo(np.int64).min
    flat_num_dates_ns_int[~nan] = flat_num_dates[~nan].astype(np.int64)
    return (pd.to_timedelta(flat_num_dates_ns_int, 'ns') + ref_date).values