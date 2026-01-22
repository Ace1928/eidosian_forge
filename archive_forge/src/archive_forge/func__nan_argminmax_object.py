from __future__ import annotations
import warnings
import numpy as np
from xarray.core import dtypes, duck_array_ops, nputils, utils
from xarray.core.duck_array_ops import (
def _nan_argminmax_object(func, fill_value, value, axis=None, **kwargs):
    """In house nanargmin, nanargmax for object arrays. Always return integer
    type
    """
    valid_count = count(value, axis=axis)
    value = fillna(value, fill_value)
    data = getattr(np, func)(value, axis=axis, **kwargs)
    if (valid_count == 0).any():
        raise ValueError('All-NaN slice encountered')
    return data