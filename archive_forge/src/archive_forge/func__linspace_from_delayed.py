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
def _linspace_from_delayed(start, stop, num=50):
    linspace_name = 'linspace-' + tokenize(start, stop, num)
    (start_ref, stop_ref, num_ref), deps = unpack_collections([start, stop, num])
    if len(deps) == 0:
        return np.linspace(start, stop, num=num)
    linspace_dsk = {(linspace_name, 0): (np.linspace, start_ref, stop_ref, num_ref)}
    linspace_graph = HighLevelGraph.from_collections(linspace_name, linspace_dsk, dependencies=deps)
    chunks = ((np.nan,),) if is_dask_collection(num) else ((num,),)
    return Array(linspace_graph, linspace_name, chunks, dtype=float)