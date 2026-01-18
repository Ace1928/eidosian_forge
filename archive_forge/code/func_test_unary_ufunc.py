from __future__ import annotations
import pickle
import warnings
from functools import partial
from operator import add
import pytest
import dask.array as da
from dask.array.ufunc import da_frompyfunc
from dask.array.utils import assert_eq
from dask.base import tokenize
@pytest.mark.parametrize('ufunc', unary_ufuncs)
def test_unary_ufunc(ufunc):
    if ufunc == 'fix':
        pytest.skip('fix calls floor in a way that we do not yet support')
    dafunc = getattr(da, ufunc)
    npfunc = getattr(np, ufunc)
    arr = np.random.randint(1, 100, size=(20, 20))
    darr = da.from_array(arr, 3)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        assert isinstance(dafunc(darr), da.Array)
        assert_eq(dafunc(darr), npfunc(arr), equal_nan=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        if isinstance(npfunc, np.ufunc):
            assert isinstance(npfunc(darr), da.Array)
        else:
            assert isinstance(npfunc(darr), np.ndarray)
        assert_eq(npfunc(darr), npfunc(arr), equal_nan=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        assert isinstance(dafunc(arr), np.ndarray)
        assert_eq(dafunc(arr), npfunc(arr), equal_nan=True)