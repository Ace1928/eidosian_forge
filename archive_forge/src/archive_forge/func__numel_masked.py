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
@numel_lookup.register(np.ma.masked_array)
def _numel_masked(x, **kwargs):
    """Numel implementation for masked arrays."""
    return chunk.sum(np.ones_like(x), **kwargs)