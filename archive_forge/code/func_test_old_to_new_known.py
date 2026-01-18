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
def test_old_to_new_known():
    old = ((10, 10, 10, 10, 10),)
    new = ((25, 5, 20),)
    result = old_to_new(old, new)
    expected = [[[(0, slice(0, 10, None)), (1, slice(0, 10, None)), (2, slice(0, 5, None))], [(2, slice(5, 10, None))], [(3, slice(0, 10, None)), (4, slice(0, 10, None))]]]
    assert result == expected