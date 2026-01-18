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
@pytest.mark.parametrize('name', ['log', 'modf', 'frexp'])
def test_ufunc_meta(name):
    disclaimer = DISCLAIMER.format(name=name)
    skip_test = '  # doctest: +SKIP'
    ufunc = getattr(da, name)
    assert ufunc.__name__ == name
    assert disclaimer in ufunc.__doc__
    assert ufunc.__doc__.replace(disclaimer, '').replace(skip_test, '') == getattr(np, name).__doc__