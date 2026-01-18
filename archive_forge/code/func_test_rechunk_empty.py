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
def test_rechunk_empty():
    x = da.ones((0, 10), chunks=(5, 5))
    y = x.rechunk((2, 2))
    assert y.chunks == ((0,), (2,) * 5)
    assert_eq(x, y)