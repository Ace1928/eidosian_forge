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
def test_out_shape_mismatch():
    x = da.arange(10, chunks=(5,))
    y = da.arange(15, chunks=(5,))
    with pytest.raises(ValueError):
        assert np.log(x, out=y)