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
def _infer_time_units_from_diff(unique_timedeltas) -> str:
    unit_timedelta: Callable[[str], timedelta] | Callable[[str], np.timedelta64]
    zero_timedelta: timedelta | np.timedelta64
    if unique_timedeltas.dtype == np.dtype('O'):
        time_units = _NETCDF_TIME_UNITS_CFTIME
        unit_timedelta = _unit_timedelta_cftime
        zero_timedelta = timedelta(microseconds=0)
    else:
        time_units = _NETCDF_TIME_UNITS_NUMPY
        unit_timedelta = _unit_timedelta_numpy
        zero_timedelta = np.timedelta64(0, 'ns')
    for time_unit in time_units:
        if np.all(unique_timedeltas % unit_timedelta(time_unit) == zero_timedelta):
            return time_unit
    return 'seconds'