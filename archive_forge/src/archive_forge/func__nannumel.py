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
@nannumel_lookup.register((object, np.ndarray))
def _nannumel(x, **kwargs):
    """A reduction to count the number of elements, excluding nans"""
    return chunk.sum(~np.isnan(x), **kwargs)