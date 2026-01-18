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
def wrap_func_like(func, *args, **kwargs):
    """
    Transform np creation function into blocked version
    """
    x = args[0]
    meta = meta_from_array(x)
    shape = kwargs.get('shape', x.shape)
    parsed = _parse_wrap_args(func, args, kwargs, shape)
    shape = parsed['shape']
    dtype = parsed['dtype']
    chunks = parsed['chunks']
    name = parsed['name']
    kwargs = parsed['kwargs']
    keys = product([name], *[range(len(bd)) for bd in chunks])
    shapes = product(*chunks)
    shapes = list(shapes)
    kw = [kwargs for _ in shapes]
    for i, s in enumerate(list(shapes)):
        kw[i]['shape'] = s
    vals = ((partial(func, dtype=dtype, **k),) + args for k, s in zip(kw, shapes))
    dsk = dict(zip(keys, vals))
    return Array(dsk, name, chunks, meta=meta.astype(dtype))