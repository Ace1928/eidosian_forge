from __future__ import annotations
import builtins
import contextlib
import math
import operator
import warnings
from collections.abc import Iterable
from functools import partial
from itertools import product, repeat
from numbers import Integral, Number
import numpy as np
from tlz import accumulate, compose, drop, get, partition_all, pluck
from dask import config
from dask.array import chunk
from dask.array.blockwise import blockwise
from dask.array.core import (
from dask.array.creation import arange, diagonal
from dask.array.dispatch import divide_lookup, nannumel_lookup, numel_lookup
from dask.array.numpy_compat import ComplexWarning
from dask.array.utils import (
from dask.array.wrap import ones, zeros
from dask.base import tokenize
from dask.blockwise import lol_tuples
from dask.highlevelgraph import HighLevelGraph
from dask.utils import (
def argtopk(a, k, axis=-1, split_every=None):
    """Extract the indices of the k largest elements from a on the given axis,
    and return them sorted from largest to smallest. If k is negative, extract
    the indices of the -k smallest elements instead, and return them sorted
    from smallest to largest.

    This performs best when ``k`` is much smaller than the chunk size. All
    results will be returned in a single chunk along the given axis.

    Parameters
    ----------
    x: Array
        Data being sorted
    k: int
    axis: int, optional
    split_every: int >=2, optional
        See :func:`topk`. The performance considerations for topk also apply
        here.

    Returns
    -------
    Selection of np.intp indices of x with size abs(k) along the given axis.

    Examples
    --------
    >>> import dask.array as da
    >>> x = np.array([5, 1, 3, 6])
    >>> d = da.from_array(x, chunks=2)
    >>> d.argtopk(2).compute()
    array([3, 0])
    >>> d.argtopk(-2).compute()
    array([1, 2])
    """
    axis = validate_axis(axis, a.ndim)
    idx = arange(a.shape[axis], chunks=(a.chunks[axis],), dtype=np.intp)
    idx = idx[tuple((slice(None) if i == axis else np.newaxis for i in range(a.ndim)))]
    a_plus_idx = a.map_blocks(chunk.argtopk_preprocess, idx, dtype=object)
    chunk_combine = partial(chunk.argtopk, k=k)
    aggregate = partial(chunk.argtopk_aggregate, k=k)
    if isinstance(axis, Number):
        naxis = 1
    else:
        naxis = len(axis)
    meta = a._meta.astype(np.intp).reshape((0,) * (a.ndim - naxis + 1))
    return reduction(a_plus_idx, chunk=chunk_combine, combine=chunk_combine, aggregate=aggregate, axis=axis, keepdims=True, dtype=np.intp, split_every=split_every, concatenate=False, output_size=abs(k), meta=meta)