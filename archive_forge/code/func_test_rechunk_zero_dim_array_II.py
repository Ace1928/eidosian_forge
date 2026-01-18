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
def test_rechunk_zero_dim_array_II():
    x = da.zeros((4, 0, 6, 10), chunks=3)
    y = x.rechunk({0: 4, 2: 2})
    assert y.chunks == ((4,), (0,), (2, 2, 2), (3, 3, 3, 1))
    assert_eq(x, y)