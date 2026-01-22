from __future__ import annotations
import math
import warnings
from collections.abc import Iterable
from functools import partial, reduce, wraps
from numbers import Integral, Real
import numpy as np
from tlz import concat, interleave, sliding_window
from dask.array import chunk
from dask.array.core import (
from dask.array.creation import arange, diag, empty, indices, tri
from dask.array.einsumfuncs import einsum  # noqa
from dask.array.numpy_compat import NUMPY_GE_200
from dask.array.reductions import reduction
from dask.array.ufunc import multiply, sqrt
from dask.array.utils import (
from dask.array.wrap import ones
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.delayed import Delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.utils import apply, derived_from, funcname, is_arraylike, is_cupy_type
def aligned_coarsen_chunks(chunks: list[int], multiple: int) -> tuple[int, ...]:
    """
    Returns a new chunking aligned with the coarsening multiple.
    Any excess is at the end of the array.

    Examples
    --------
    >>> aligned_coarsen_chunks(chunks=(1, 2, 3), multiple=4)
    (4, 2)
    >>> aligned_coarsen_chunks(chunks=(1, 20, 3, 4), multiple=4)
    (4, 20, 4)
    >>> aligned_coarsen_chunks(chunks=(20, 10, 15, 23, 24), multiple=10)
    (20, 10, 20, 20, 20, 2)
    """
    overflow = np.array(chunks) % multiple
    excess = overflow.sum()
    new_chunks = np.array(chunks) - overflow
    chunk_validity = new_chunks == chunks
    valid_inds, invalid_inds = (np.where(chunk_validity)[0], np.where(~chunk_validity)[0])
    chunk_modification_order = [*invalid_inds[np.argsort(new_chunks[invalid_inds])], *valid_inds[np.argsort(new_chunks[valid_inds])]]
    partitioned_excess, remainder = _partition(excess, multiple)
    for idx, extra in enumerate(partitioned_excess):
        new_chunks[chunk_modification_order[idx]] += extra
    new_chunks = np.array([*new_chunks, *remainder])
    new_chunks = new_chunks[new_chunks > 0]
    return tuple(new_chunks)