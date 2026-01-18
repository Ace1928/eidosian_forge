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
def test_weighted_reduction():

    def w_sum(x, weights=None, dtype=None, computing_meta=False, **kwargs):
        """`chunk` callable for (weighted) sum"""
        if computing_meta:
            return x
        if weights is not None:
            x = x * weights
        return np.sum(x, dtype=dtype, **kwargs)
    a = 1 + np.ma.arange(60).reshape(6, 10)
    a[2, 2] = np.ma.masked
    dx = da.from_array(a, chunks=(4, 5))
    w = np.linspace(1, 2, 6).reshape(6, 1)
    x = da.reduction(dx, w_sum, np.sum, dtype=dx.dtype)
    assert_eq(x, np.sum(a), check_shape=True)
    x = da.reduction(dx, w_sum, np.sum, dtype='f8', weights=w)
    assert_eq(x, np.sum(a * w), check_shape=True)
    with pytest.raises(ValueError):
        da.reduction(dx, w_sum, np.sum, weights=[1, 2, 3])
    with pytest.raises(ValueError):
        da.reduction(dx, w_sum, np.sum, weights=[[[2]]])