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
def _nanmax_skip(x_chunk, axis, keepdims):
    if x_chunk.size > 0:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
            return np.nanmax(x_chunk, axis=axis, keepdims=keepdims)
    else:
        return asarray_safe(np.array([], dtype=x_chunk.dtype), like=meta_from_array(x_chunk))