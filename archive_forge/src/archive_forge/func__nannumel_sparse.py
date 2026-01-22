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
def _nannumel_sparse(x, **kwargs):
    """
    A reduction to count the number of elements in a sparse array, excluding nans.
    This will in general result in a dense matrix with an unpredictable fill value.
    So make it official and convert it to dense.

    https://github.com/dask/dask/issues/7169
    """
    n = _nannumel(x, **kwargs)
    return n.todense() if hasattr(n, 'todense') else n