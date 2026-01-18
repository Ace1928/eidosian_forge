from __future__ import annotations
import os
import warnings
from contextlib import nullcontext as does_not_warn
from itertools import permutations, zip_longest
import pytest
import itertools
import dask.array as da
import dask.config as config
from dask.array.numpy_compat import NUMPY_GE_122, ComplexWarning
from dask.array.utils import assert_eq, same_keys
from dask.core import get_deps
def test_tree_reduce_set_options():
    x = da.from_array(np.arange(242).reshape((11, 22)), chunks=(3, 4))
    with config.set(split_every={0: 2, 1: 3}):
        assert_max_deps(x.sum(), 2 * 3)
        assert_max_deps(x.sum(axis=()), 1)
        assert_max_deps(x.sum(axis=0), 2)