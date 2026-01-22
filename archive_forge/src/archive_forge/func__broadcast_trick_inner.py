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
@curry
def _broadcast_trick_inner(func, shape, meta=(), *args, **kwargs):
    null_shape = () if shape == () else 1
    return np.broadcast_to(func(meta, *args, shape=null_shape, **kwargs), shape)