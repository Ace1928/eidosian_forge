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
def _should_cftime_be_used(source, target_calendar: str, use_cftime: bool | None) -> bool:
    """Return whether conversion of the source to the target calendar should
    result in a cftime-backed array.

    Source is a 1D datetime array, target_cal a string (calendar name) and
    use_cftime is a boolean or None. If use_cftime is None, this returns True
    if the source's range and target calendar are convertible to np.datetime64 objects.
    """
    if use_cftime is not True:
        if _is_standard_calendar(target_calendar):
            if _is_numpy_compatible_time_range(source):
                return False
            elif use_cftime is False:
                raise ValueError('Source time range is not valid for numpy datetimes. Try using `use_cftime=True`.')
        elif use_cftime is False:
            raise ValueError(f"Calendar '{target_calendar}' is only valid with cftime. Try using `use_cftime=True`.")
    return True