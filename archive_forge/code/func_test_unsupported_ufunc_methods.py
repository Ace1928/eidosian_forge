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
def test_unsupported_ufunc_methods():
    x = da.arange(10, chunks=(5,))
    with pytest.raises(TypeError):
        assert np.add.reduce(x)