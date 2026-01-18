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
def wrap_func_shape_as_first_arg(func, *args, **kwargs):
    """
    Transform np creation function into blocked version
    """
    if 'shape' not in kwargs:
        shape, args = (args[0], args[1:])
    else:
        shape = kwargs.pop('shape')
    if isinstance(shape, Array):
        raise TypeError('Dask array input not supported. Please use tuple, list, or a 1D numpy array instead.')
    parsed = _parse_wrap_args(func, args, kwargs, shape)
    shape = parsed['shape']
    dtype = parsed['dtype']
    chunks = parsed['chunks']
    name = parsed['name']
    kwargs = parsed['kwargs']
    func = partial(func, dtype=dtype, **kwargs)
    out_ind = dep_ind = tuple(range(len(shape)))
    graph = core_blockwise(func, name, out_ind, ArrayChunkShapeDep(chunks), dep_ind, numblocks={})
    return Array(graph, name, chunks, dtype=dtype, meta=kwargs.get('meta', None))