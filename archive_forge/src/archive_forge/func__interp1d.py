from __future__ import annotations
import datetime as dt
import warnings
from collections.abc import Hashable, Sequence
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, get_args
import numpy as np
import pandas as pd
from xarray.core import utils
from xarray.core.common import _contains_datetime_like_objects, ones_like
from xarray.core.computation import apply_ufunc
from xarray.core.duck_array_ops import (
from xarray.core.options import _get_keep_attrs
from xarray.core.types import Interp1dOptions, InterpOptions
from xarray.core.utils import OrderedSet, is_scalar
from xarray.core.variable import Variable, broadcast_variables
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def _interp1d(var, x, new_x, func, kwargs):
    x, new_x = (x[0], new_x[0])
    rslt = func(x, var, assume_sorted=True, **kwargs)(np.ravel(new_x))
    if new_x.ndim > 1:
        return reshape(rslt, var.shape[:-1] + new_x.shape)
    if new_x.ndim == 0:
        return rslt[..., -1]
    return rslt