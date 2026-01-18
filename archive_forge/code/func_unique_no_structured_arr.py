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
def unique_no_structured_arr(ar, return_index=False, return_inverse=False, return_counts=False):
    if return_index is not False or return_inverse is not False or return_counts is not False:
        raise ValueError("dask.array.unique does not support `return_index`, `return_inverse` or `return_counts` with array types that don't support structured arrays.")
    ar = ar.ravel()
    args = [ar, 'i']
    meta = meta_from_array(ar)
    out = blockwise(np.unique, 'i', *args, meta=meta)
    out._chunks = tuple(((np.nan,) * len(c) for c in out.chunks))
    out_parts = [out]
    name = 'unique-aggregate-' + out.name
    dsk = {(name, 0): (np.unique,) + tuple(((np.concatenate, o.__dask_keys__()) if hasattr(o, '__dask_keys__') else o for o in out_parts))}
    dependencies = [o for o in out_parts if hasattr(o, '__dask_keys__')]
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=dependencies)
    chunks = ((np.nan,),)
    out = Array(graph, name, chunks, meta=meta)
    result = [out]
    if len(result) == 1:
        result = result[0]
    else:
        result = tuple(result)
    return result