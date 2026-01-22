from __future__ import annotations
import contextlib
import math
import operator
import os
import pickle
import re
import sys
import traceback
import uuid
import warnings
from bisect import bisect
from collections.abc import (
from functools import partial, reduce, wraps
from itertools import product, zip_longest
from numbers import Integral, Number
from operator import add, mul
from threading import Lock
from typing import Any, TypeVar, Union, cast
import numpy as np
from numpy.typing import ArrayLike
from tlz import accumulate, concat, first, frequencies, groupby, partition
from tlz.curried import pluck
from dask import compute, config, core
from dask.array import chunk
from dask.array.chunk import getitem
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.dispatch import (  # noqa: F401
from dask.array.numpy_compat import _Recurser
from dask.array.slicing import replace_ellipsis, setitem_array, slice_array
from dask.base import (
from dask.blockwise import blockwise as core_blockwise
from dask.blockwise import broadcast_dimensions
from dask.context import globalmethod
from dask.core import quote
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.layers import ArraySliceDep, reshapelist
from dask.sizeof import sizeof
from dask.typing import Graph, Key, NestedKeys
from dask.utils import (
from dask.widgets import get_template
from dask.array.optimization import fuse_slice, optimize
from dask.array.blockwise import blockwise
from dask.array.utils import compute_meta, meta_from_array
def _vindex_array(x, dict_indexes):
    """Point wise indexing with only NumPy Arrays."""
    try:
        broadcast_indexes = np.broadcast_arrays(*dict_indexes.values())
    except ValueError as e:
        shapes_str = ' '.join((str(a.shape) for a in dict_indexes.values()))
        raise IndexError('shape mismatch: indexing arrays could not be broadcast together with shapes ' + shapes_str) from e
    broadcast_shape = broadcast_indexes[0].shape
    lookup = dict(zip(dict_indexes, broadcast_indexes))
    flat_indexes = [lookup[i].ravel().tolist() if i in lookup else None for i in range(x.ndim)]
    flat_indexes.extend([None] * (x.ndim - len(flat_indexes)))
    flat_indexes = [list(index) if index is not None else index for index in flat_indexes]
    bounds = [list(accumulate(add, (0,) + c)) for c in x.chunks]
    bounds2 = [b for i, b in zip(flat_indexes, bounds) if i is not None]
    axis = _get_axis(flat_indexes)
    token = tokenize(x, flat_indexes)
    out_name = 'vindex-merge-' + token
    points = list()
    for i, idx in enumerate(zip(*[i for i in flat_indexes if i is not None])):
        block_idx = [bisect(b, ind) - 1 for b, ind in zip(bounds2, idx)]
        inblock_idx = [ind - bounds2[k][j] for k, (ind, j) in enumerate(zip(idx, block_idx))]
        points.append((i, tuple(block_idx), tuple(inblock_idx)))
    chunks = [c for i, c in zip(flat_indexes, x.chunks) if i is None]
    chunks.insert(0, (len(points),) if points else (0,))
    chunks = tuple(chunks)
    if points:
        per_block = groupby(1, points)
        per_block = {k: v for k, v in per_block.items() if v}
        other_blocks = list(product(*[list(range(len(c))) if i is None else [None] for i, c in zip(flat_indexes, x.chunks)]))
        full_slices = [slice(None, None) if i is None else None for i in flat_indexes]
        name = 'vindex-slice-' + token
        vindex_merge_name = 'vindex-merge-' + token
        dsk = {}
        for okey in other_blocks:
            for i, key in enumerate(per_block):
                dsk[keyname(name, i, okey)] = (_vindex_transpose, (_vindex_slice, (x.name,) + interleave_none(okey, key), interleave_none(full_slices, list(zip(*pluck(2, per_block[key]))))), axis)
            dsk[keyname(vindex_merge_name, 0, okey)] = (_vindex_merge, [list(pluck(0, per_block[key])) for key in per_block], [keyname(name, i, okey) for i in range(len(per_block))])
        result_1d = Array(HighLevelGraph.from_collections(out_name, dsk, dependencies=[x]), out_name, chunks, x.dtype, meta=x._meta)
        return result_1d.reshape(broadcast_shape + result_1d.shape[1:])
    from dask.array.wrap import empty
    result_1d = empty(tuple(map(sum, chunks)), chunks=chunks, dtype=x.dtype, name=out_name)
    return result_1d.reshape(broadcast_shape + result_1d.shape[1:])