from __future__ import annotations
import heapq
import math
from functools import reduce
from itertools import chain, count, product
from operator import add, itemgetter, mul
from warnings import warn
import numpy as np
import tlz as toolz
from tlz import accumulate
from dask import config
from dask.array.chunk import getitem
from dask.array.core import Array, concatenate3, normalize_chunks
from dask.array.utils import validate_axis
from dask.array.wrap import empty
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import parse_bytes
def _compute_rechunk(x, chunks):
    """Compute the rechunk of *x* to the given *chunks*."""
    if x.size == 0:
        return empty(x.shape, chunks=chunks, dtype=x.dtype)
    ndim = x.ndim
    crossed = intersect_chunks(x.chunks, chunks)
    x2 = dict()
    intermediates = dict()
    token = tokenize(x, chunks)
    merge_name = 'rechunk-merge-' + token
    split_name = 'rechunk-split-' + token
    split_name_suffixes = count()
    old_blocks = np.empty([len(c) for c in x.chunks], dtype='O')
    for index in np.ndindex(old_blocks.shape):
        old_blocks[index] = (x.name,) + index
    new_index = product(*(range(len(c)) for c in chunks))
    for new_idx, cross1 in zip(new_index, crossed):
        key = (merge_name,) + new_idx
        old_block_indices = [[cr[i][0] for cr in cross1] for i in range(ndim)]
        subdims1 = [len(set(old_block_indices[i])) for i in range(ndim)]
        rec_cat_arg = np.empty(subdims1, dtype='O')
        rec_cat_arg_flat = rec_cat_arg.flat
        for rec_cat_index, ind_slices in enumerate(cross1):
            old_block_index, slices = zip(*ind_slices)
            name = (split_name, next(split_name_suffixes))
            old_index = old_blocks[old_block_index][1:]
            if all((slc.start == 0 and slc.stop == x.chunks[i][ind] for i, (slc, ind) in enumerate(zip(slices, old_index)))):
                rec_cat_arg_flat[rec_cat_index] = old_blocks[old_block_index]
            else:
                intermediates[name] = (getitem, old_blocks[old_block_index], slices)
                rec_cat_arg_flat[rec_cat_index] = name
        assert rec_cat_index == rec_cat_arg.size - 1
        if all((d == 1 for d in rec_cat_arg.shape)):
            x2[key] = rec_cat_arg.flat[0]
        else:
            x2[key] = (concatenate3, rec_cat_arg.tolist())
    del old_blocks, new_index
    layer = toolz.merge(x2, intermediates)
    graph = HighLevelGraph.from_collections(merge_name, layer, dependencies=[x])
    return Array(graph, merge_name, chunks, meta=x)