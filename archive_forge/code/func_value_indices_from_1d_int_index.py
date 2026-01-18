from __future__ import annotations
import bisect
import functools
import math
import warnings
from itertools import product
from numbers import Integral, Number
from operator import itemgetter
import numpy as np
from tlz import concat, memoize, merge, pluck
from dask import config, core, utils
from dask.array.chunk import getitem
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import cached_cumsum, is_arraylike
@functools.lru_cache
def value_indices_from_1d_int_index(dim, vsize, loc0, loc1):
    """Value indices for index elements between loc0 and loc1.

        The index is the input assignment index that is defined in the
        namespace of the caller. It is assumed that negative elements
        have already been posified.

        Parameters
        ----------
        dim : `int`
           The dimension position of the index that is used as a proxy
           for the non-hashable index to define the LRU cache key.
        vsize : `int`
            The full size of the dimension of the assignment value.
        loc0 : `int`
            The start index of the block along the dimension.
        loc1 : `int`
            The stop index of the block along the dimension.

        Returns
        -------
        numpy array or dask array
            If index is a numpy array then a numpy array is
            returned.

            If index is dask array then a dask array is returned.

        """
    if is_dask_collection(index):
        if np.isnan(index.size):
            i = np.where((loc0 <= index) & (index < loc1), True, False)
            i = concatenate_array_chunks(i)
            i._chunks = ((vsize,),)
        else:
            i = np.where((loc0 <= index) & (index < loc1))[0]
            i = concatenate_array_chunks(i)
    else:
        i = np.where((loc0 <= index) & (index < loc1))[0]
    return i