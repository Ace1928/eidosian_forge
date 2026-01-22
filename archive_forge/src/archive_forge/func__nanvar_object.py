from __future__ import annotations
import warnings
import numpy as np
from xarray.core import dtypes, duck_array_ops, nputils, utils
from xarray.core.duck_array_ops import (
def _nanvar_object(value, axis=None, ddof=0, keepdims=False, **kwargs):
    value_mean = _nanmean_ddof_object(ddof=0, value=value, axis=axis, keepdims=True, **kwargs)
    squared = (astype(value, value_mean.dtype) - value_mean) ** 2
    return _nanmean_ddof_object(ddof, squared, axis=axis, keepdims=keepdims, **kwargs)