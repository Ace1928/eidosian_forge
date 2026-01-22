from __future__ import annotations
import math
import numpy as np
from dask.array import chunk
from dask.array.core import Array
from dask.array.dispatch import (
from dask.array.numpy_compat import divide as np_divide
from dask.array.numpy_compat import ma_divide
from dask.array.percentile import _percentile
from dask.backends import CreationDispatch, DaskBackendEntrypoint
def _numel(x, coerce_np_ndarray: bool, **kwargs):
    """
    A reduction to count the number of elements.

    This has an additional kwarg in coerce_np_ndarray, which determines
    whether to ensure that the resulting array is a numpy.ndarray, or whether
    we allow it to be other array types via `np.full_like`.
    """
    shape = x.shape
    keepdims = kwargs.get('keepdims', False)
    axis = kwargs.get('axis', None)
    dtype = kwargs.get('dtype', np.float64)
    if axis is None:
        prod = np.prod(shape, dtype=dtype)
        if keepdims is False:
            return prod
        if coerce_np_ndarray:
            return np.full(shape=(1,) * len(shape), fill_value=prod, dtype=dtype)
        else:
            return np.full_like(x, prod, shape=(1,) * len(shape), dtype=dtype)
    if not isinstance(axis, (tuple, list)):
        axis = [axis]
    prod = math.prod((shape[dim] for dim in axis))
    if keepdims is True:
        new_shape = tuple((shape[dim] if dim not in axis else 1 for dim in range(len(shape))))
    else:
        new_shape = tuple((shape[dim] for dim in range(len(shape)) if dim not in axis))
    if coerce_np_ndarray:
        return np.broadcast_to(np.array(prod, dtype=dtype), new_shape)
    else:
        return np.full_like(x, prod, shape=new_shape, dtype=dtype)