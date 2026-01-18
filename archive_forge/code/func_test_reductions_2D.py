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
@pytest.mark.slow
@pytest.mark.parametrize('dtype', ['f4', 'i4', 'c8'])
def test_reductions_2D(dtype):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ComplexWarning)
        x = (np.arange(1, 122) + 1j * np.arange(1, 122)).reshape((11, 11)).astype(dtype)
    a = da.from_array(x, chunks=(4, 4))
    b = a.sum(keepdims=True)
    assert b.__dask_keys__() == [[(b.name, 0, 0)]]
    reduction_2d_test(da.sum, a, np.sum, x)
    reduction_2d_test(da.mean, a, np.mean, x)
    reduction_2d_test(da.var, a, np.var, x, False)
    reduction_2d_test(da.std, a, np.std, x, False)
    reduction_2d_test(da.min, a, np.min, x, False)
    reduction_2d_test(da.max, a, np.max, x, False)
    reduction_2d_test(da.any, a, np.any, x, False)
    reduction_2d_test(da.all, a, np.all, x, False)
    reduction_2d_test(da.nansum, a, np.nansum, x)
    reduction_2d_test(da.nanmean, a, np.mean, x)
    reduction_2d_test(da.nanvar, a, np.nanvar, x, False)
    reduction_2d_test(da.nanstd, a, np.nanstd, x, False)
    reduction_2d_test(da.nanmin, a, np.nanmin, x, False)
    reduction_2d_test(da.nanmax, a, np.nanmax, x, False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        reduction_2d_test(da.prod, a, np.prod, x)
        reduction_2d_test(da.nanprod, a, np.nanprod, x)