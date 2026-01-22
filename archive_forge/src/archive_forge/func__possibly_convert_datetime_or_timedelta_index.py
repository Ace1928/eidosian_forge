from __future__ import annotations
import copy
import itertools
import math
import numbers
import warnings
from collections.abc import Hashable, Mapping, Sequence
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, NoReturn, cast
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import xarray as xr  # only for Dataset and DataArray
from xarray.core import common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from xarray.core.arithmetic import VariableArithmetic
from xarray.core.common import AbstractArray
from xarray.core.indexing import (
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import NamedArray, _raise_if_any_duplicate_dimensions
from xarray.namedarray.pycompat import integer_types, is_0d_dask_array, to_duck_array
def _possibly_convert_datetime_or_timedelta_index(data):
    """For the time being, convert any non-nanosecond precision DatetimeIndex or
    TimedeltaIndex objects to nanosecond precision.  While pandas is relaxing
    this in version 2.0.0, in xarray we will need to make sure we are ready to
    handle non-nanosecond precision datetimes or timedeltas in our code
    before allowing such values to pass through unchanged."""
    if isinstance(data, PandasIndexingAdapter):
        if isinstance(data.array, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            data = PandasIndexingAdapter(_as_nanosecond_precision(data.array))
    elif isinstance(data, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        data = _as_nanosecond_precision(data)
    return data