from __future__ import annotations
from functools import partial
from itertools import product
import numpy as np
from tlz import curry
from dask.array.backends import array_creation_dispatch
from dask.array.core import Array, normalize_chunks
from dask.array.utils import meta_from_array
from dask.base import tokenize
from dask.blockwise import blockwise as core_blockwise
from dask.layers import ArrayChunkShapeDep
from dask.utils import funcname

    Provide a decorator to wrap common numpy function with a broadcast trick.

    Dask arrays are currently immutable; thus when we know an array is uniform,
    we can replace the actual data by a single value and have all elements point
    to it, thus reducing the size.

    >>> x = np.broadcast_to(1, (100,100,100))
    >>> x.base.nbytes
    8

    Those array are not only more efficient locally, but dask serialisation is
    aware of the _real_ size of those array and thus can send them around
    efficiently and schedule accordingly.

    Note that those array are read-only and numpy will refuse to assign to them,
    so should be safe.
    