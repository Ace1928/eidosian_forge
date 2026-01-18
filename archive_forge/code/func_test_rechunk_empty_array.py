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
@pytest.mark.parametrize('arr', [da.array([]), da.array([[], []]), da.array([[[]], [[]]])])
def test_rechunk_empty_array(arr):
    arr.rechunk()
    assert arr.size == 0