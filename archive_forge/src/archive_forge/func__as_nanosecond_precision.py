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
def _as_nanosecond_precision(data):
    dtype = data.dtype
    non_ns_datetime64 = dtype.kind == 'M' and isinstance(dtype, np.dtype) and (dtype != np.dtype('datetime64[ns]'))
    non_ns_datetime_tz_dtype = isinstance(dtype, pd.DatetimeTZDtype) and dtype.unit != 'ns'
    if non_ns_datetime64 or non_ns_datetime_tz_dtype:
        utils.emit_user_level_warning(NON_NANOSECOND_WARNING.format(case='datetime'))
        if isinstance(dtype, pd.DatetimeTZDtype):
            nanosecond_precision_dtype = pd.DatetimeTZDtype('ns', dtype.tz)
        else:
            nanosecond_precision_dtype = 'datetime64[ns]'
        return duck_array_ops.astype(data, nanosecond_precision_dtype)
    elif dtype.kind == 'm' and dtype != np.dtype('timedelta64[ns]'):
        utils.emit_user_level_warning(NON_NANOSECOND_WARNING.format(case='timedelta'))
        return duck_array_ops.astype(data, 'timedelta64[ns]')
    else:
        return data