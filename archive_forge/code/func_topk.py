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
def topk(a, k, axis=-1, split_every=None):
    """Extract the k largest elements from a on the given axis,
    and return them sorted from largest to smallest.
    If k is negative, extract the -k smallest elements instead,
    and return them sorted from smallest to largest.

    This performs best when ``k`` is much smaller than the chunk size. All
    results will be returned in a single chunk along the given axis.

    Parameters
    ----------
    x: Array
        Data being sorted
    k: int
    axis: int, optional
    split_every: int >=2, optional
        See :func:`reduce`. This parameter becomes very important when k is
        on the same order of magnitude of the chunk size or more, as it
        prevents getting the whole or a significant portion of the input array
        in memory all at once, with a negative impact on network transfer
        too when running on distributed.

    Returns
    -------
    Selection of x with size abs(k) along the given axis.

    Examples
    --------
    >>> import dask.array as da
    >>> x = np.array([5, 1, 3, 6])
    >>> d = da.from_array(x, chunks=2)
    >>> d.topk(2).compute()
    array([6, 5])
    >>> d.topk(-2).compute()
    array([1, 3])
    """
    axis = validate_axis(axis, a.ndim)
    chunk_combine = partial(chunk.topk, k=k)
    aggregate = partial(chunk.topk_aggregate, k=k)
    return reduction(a, chunk=chunk_combine, combine=chunk_combine, aggregate=aggregate, axis=axis, keepdims=True, dtype=a.dtype, split_every=split_every, output_size=abs(k))