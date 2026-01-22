from __future__ import annotations
import warnings
import numpy as np
from xarray.core import dtypes, duck_array_ops, nputils, utils
from xarray.core.duck_array_ops import (
def _nanmean_ddof_object(ddof, value, axis=None, dtype=None, **kwargs):
    """In house nanmean. ddof argument will be used in _nanvar method"""
    from xarray.core.duck_array_ops import count, fillna, where_method
    valid_count = count(value, axis=axis)
    value = fillna(value, 0)
    if dtype is None and value.dtype.kind == 'O':
        dtype = value.dtype if value.dtype.kind in ['cf'] else float
    data = np.sum(value, axis=axis, dtype=dtype, **kwargs)
    data = data / (valid_count - ddof)
    return where_method(data, valid_count != 0)