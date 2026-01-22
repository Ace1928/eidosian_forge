from __future__ import annotations
from functools import partial
from itertools import product
import numpy as np
from tlz import curry
from dask.array.backends import array_creation_dispatch
from dask.array.core import Array, normalize_chunks
from dask.array.utils import meta_from_array
from dask.base import tokenize
from dask.blockwise import blockwise as core_blockwise
from dask.layers import ArrayChunkShapeDep
from dask.utils import funcname
def _parse_wrap_args(func, args, kwargs, shape):
    if isinstance(shape, np.ndarray):
        shape = shape.tolist()
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    name = kwargs.pop('name', None)
    chunks = kwargs.pop('chunks', 'auto')
    dtype = kwargs.pop('dtype', None)
    if dtype is None:
        dtype = func(shape, *args, **kwargs).dtype
    dtype = np.dtype(dtype)
    chunks = normalize_chunks(chunks, shape, dtype=dtype)
    name = name or funcname(func) + '-' + tokenize(func, shape, chunks, dtype, args, kwargs)
    return {'shape': shape, 'dtype': dtype, 'kwargs': kwargs, 'chunks': chunks, 'name': name}