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
def arange_safe(*args, like, **kwargs):
    """
    Use the `like=` from `np.arange` to create a new array dispatching
    to the downstream library. If that fails, falls back to the
    default NumPy behavior, resulting in a `numpy.ndarray`.
    """
    if like is None:
        return np.arange(*args, **kwargs)
    else:
        try:
            return np.arange(*args, like=meta_from_array(like), **kwargs)
        except TypeError:
            return np.arange(*args, **kwargs)