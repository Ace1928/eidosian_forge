from __future__ import annotations
import warnings
from numbers import Integral, Number
import numpy as np
from tlz import concat, get, partial
from tlz.curried import map
from dask.array import chunk
from dask.array.core import Array, concatenate, map_blocks, unify_chunks
from dask.array.creation import empty_like, full_like
from dask.array.numpy_compat import normalize_axis_tuple
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayOverlapLayer
from dask.utils import derived_from
def trim_internal(x, axes, boundary=None):
    """Trim sides from each block

    This couples well with the overlap operation, which may leave excess data on
    each block

    See also
    --------
    dask.array.chunk.trim
    dask.array.map_blocks
    """
    boundary = coerce_boundary(x.ndim, boundary)
    olist = []
    for i, bd in enumerate(x.chunks):
        bdy = boundary.get(i, 'none')
        overlap = axes.get(i, 0)
        ilist = []
        for j, d in enumerate(bd):
            if bdy != 'none':
                if isinstance(overlap, tuple):
                    d = d - sum(overlap)
                else:
                    d = d - overlap * 2
            elif isinstance(overlap, tuple):
                d = d - overlap[0] if j != 0 else d
                d = d - overlap[1] if j != len(bd) - 1 else d
            else:
                d = d - overlap if j != 0 else d
                d = d - overlap if j != len(bd) - 1 else d
            ilist.append(d)
        olist.append(tuple(ilist))
    chunks = tuple(olist)
    return map_blocks(partial(_trim, axes=axes, boundary=boundary), x, chunks=chunks, dtype=x.dtype, meta=x._meta)