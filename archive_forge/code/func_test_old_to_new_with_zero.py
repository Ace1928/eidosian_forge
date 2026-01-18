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
def test_old_to_new_with_zero():
    from dask.array.rechunk import old_to_new
    old = ((4, 4),)
    new = ((4, 0, 4),)
    result = old_to_new(old, new)
    expected = [[[(0, slice(0, 4))], [(1, slice(0, 0))], [(1, slice(0, 4))]]]
    assert result == expected
    old = ((4,),)
    new = ((4, 0),)
    result = old_to_new(old, new)
    expected = [[[(0, slice(0, 4))], [(0, slice(4, 4))]]]
    assert result == expected
    old = ((4, 0, 4),)
    new = ((4, 0, 2, 2),)
    result = old_to_new(old, new)
    expected = [[[(0, slice(0, 4))], [(2, slice(0, 0))], [(2, slice(0, 2))], [(2, slice(2, 4))]]]
    assert result == expected