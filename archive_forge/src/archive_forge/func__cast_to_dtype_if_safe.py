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
def _cast_to_dtype_if_safe(num: np.ndarray, dtype: np.dtype) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='overflow')
        cast_num = np.asarray(num, dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        if not (num == cast_num).all():
            if np.issubdtype(num.dtype, np.floating):
                raise ValueError(f'Not possible to cast all encoded times from {num.dtype!r} to {dtype!r} without losing precision. Consider modifying the units such that integer values can be used, or removing the units and dtype encoding, at which point xarray will make an appropriate choice.')
            else:
                raise OverflowError(f'Not possible to cast encoded times from {num.dtype!r} to {dtype!r} without overflow. Consider removing the dtype encoding, at which point xarray will make an appropriate choice, or explicitly switching to a larger integer dtype.')
    elif np.isinf(cast_num).any():
        raise OverflowError(f'Not possible to cast encoded times from {num.dtype!r} to {dtype!r} without overflow.  Consider removing the dtype encoding, at which point xarray will make an appropriate choice, or explicitly switching to a larger floating point dtype.')
    return cast_num