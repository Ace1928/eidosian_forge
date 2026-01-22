from __future__ import annotations
from functools import wraps
import numpy as np
from dask.array import chunk
from dask.array.core import asanyarray, blockwise, elemwise, map_blocks
from dask.array.reductions import reduction
from dask.array.routines import _average
from dask.array.routines import nonzero as _nonzero
from dask.base import normalize_token
from dask.utils import derived_from
def _set_fill_value(x, fill_value):
    if isinstance(x, np.ma.masked_array):
        x = x.copy()
        np.ma.set_fill_value(x, fill_value=fill_value)
    return x