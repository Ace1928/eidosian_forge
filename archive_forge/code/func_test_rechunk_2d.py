from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
def test_rechunk_2d():
    """Try rechunking a random 2d matrix"""
    a = np.random.default_rng().uniform(0, 1, 300).reshape((10, 30))
    x = da.from_array(a, chunks=((1, 2, 3, 4), (5,) * 6))
    new = ((5, 5), (15,) * 2)
    x2 = rechunk(x, chunks=new)
    assert x2.chunks == new
    assert np.all(x2.compute() == a)