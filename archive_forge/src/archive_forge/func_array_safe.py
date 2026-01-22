from __future__ import annotations
import contextlib
import functools
import itertools
import math
import numbers
import warnings
import numpy as np
from tlz import concat, frequencies
from dask.array.core import Array
from dask.array.numpy_compat import AxisError
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import has_keyword, is_arraylike, is_cupy_type, typename
def array_safe(a, like, **kwargs):
    """
    If `a` is `dask.array`, return `dask.array.asarray(a, **kwargs)`,
    otherwise return `np.asarray(a, like=like, **kwargs)`, dispatching
    the call to the library that implements the like array. Note that
    when `a` is a `dask.Array` backed by `cupy.ndarray` but `like`
    isn't, this function will call `a.compute(scheduler="sync")`
    before `np.array`, as downstream libraries are unlikely to know how
    to convert a `dask.Array` and CuPy doesn't implement `__array__` to
    prevent implicit copies to host.
    """
    from dask.array.routines import array
    return _array_like_safe(np.array, array, a, like, **kwargs)