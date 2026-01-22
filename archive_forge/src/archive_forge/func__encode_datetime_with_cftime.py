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
def _encode_datetime_with_cftime(dates, units: str, calendar: str) -> np.ndarray:
    """Fallback method for encoding dates using cftime.

    This method is more flexible than xarray's parsing using datetime64[ns]
    arrays but also slower because it loops over each element.
    """
    if cftime is None:
        raise ModuleNotFoundError("No module named 'cftime'")
    if np.issubdtype(dates.dtype, np.datetime64):
        dates = dates.astype('M8[us]').astype(datetime)

    def encode_datetime(d):
        try:
            return np.nan if d is None else cftime.date2num(d, units, calendar, longdouble=False)
        except TypeError:
            return np.nan if d is None else cftime.date2num(d, units, calendar)
    return np.array([encode_datetime(d) for d in dates.ravel()]).reshape(dates.shape)