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
def _unique_internal(ar, indices, counts, return_inverse=False):
    """
    Helper/wrapper function for :func:`numpy.unique`.

    Uses :func:`numpy.unique` to find the unique values for the array chunk.
    Given this chunk may not represent the whole array, also take the
    ``indices`` and ``counts`` that are in 1-to-1 correspondence to ``ar``
    and reduce them in the same fashion as ``ar`` is reduced. Namely sum
    any counts that correspond to the same value and take the smallest
    index that corresponds to the same value.

    To handle the inverse mapping from the unique values to the original
    array, simply return a NumPy array created with ``arange`` with enough
    values to correspond 1-to-1 to the unique values. While there is more
    work needed to be done to create the full inverse mapping for the
    original array, this provides enough information to generate the
    inverse mapping in Dask.

    Given Dask likes to have one array returned from functions like
    ``blockwise``, some formatting is done to stuff all of the resulting arrays
    into one big NumPy structured array. Dask is then able to handle this
    object and can split it apart into the separate results on the Dask side,
    which then can be passed back to this function in concatenated chunks for
    further reduction or can be return to the user to perform other forms of
    analysis.

    By handling the problem in this way, it does not matter where a chunk
    is in a larger array or how big it is. The chunk can still be computed
    on the same way. Also it does not matter if the chunk is the result of
    other chunks being run through this function multiple times. The end
    result will still be just as accurate using this strategy.
    """
    return_index = indices is not None
    return_counts = counts is not None
    u = np.unique(ar)
    dt = [('values', u.dtype)]
    if return_index:
        dt.append(('indices', np.intp))
    if return_inverse:
        dt.append(('inverse', np.intp))
    if return_counts:
        dt.append(('counts', np.intp))
    r = np.empty(u.shape, dtype=dt)
    r['values'] = u
    if return_inverse:
        r['inverse'] = np.arange(len(r), dtype=np.intp)
    if return_index or return_counts:
        for i, v in enumerate(r['values']):
            m = ar == v
            if return_index:
                indices[m].min(keepdims=True, out=r['indices'][i:i + 1])
            if return_counts:
                counts[m].sum(keepdims=True, out=r['counts'][i:i + 1])
    return r