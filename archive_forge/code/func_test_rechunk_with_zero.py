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
def test_rechunk_with_zero():
    a = da.ones((8, 8), chunks=(4, 4))
    result = a.rechunk(((4, 4), (4, 0, 0, 4)))
    expected = da.ones((8, 8), chunks=((4, 4), (4, 0, 0, 4)))
    a, expected = (expected, a)
    result = a.rechunk((4, 4))
    assert_eq(result, expected)