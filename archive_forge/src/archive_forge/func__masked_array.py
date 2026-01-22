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
def _masked_array(data, mask=np.ma.nomask, masked_dtype=None, **kwargs):
    if 'chunks' in kwargs:
        del kwargs['chunks']
    return np.ma.masked_array(data, mask=mask, dtype=masked_dtype, **kwargs)