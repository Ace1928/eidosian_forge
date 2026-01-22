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
def _block_histogramdd_multiarg(*args):
    """Call numpy.histogramdd for a multi argument blocked/chunked calculation.

    Slurps the result into an additional outer axis; this new axis
    will be used to stack chunked calls of the numpy function and add
    them together later.

    The last three arguments _must be_ (bins, range, weights).

    The difference between this function and
    _block_histogramdd_rect is that here we expect the sample
    to be composed of multiple arguments (multiple 1D arrays, each one
    representing a coordinate), while _block_histogramdd_rect
    expects a single rectangular (2D array where columns are
    coordinates) sample.

    """
    bins, range, weights = args[-3:]
    sample = args[:-3]
    return np.histogramdd(sample, bins=bins, range=range, weights=weights)[0:1]