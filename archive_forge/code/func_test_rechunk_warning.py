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
def test_rechunk_warning():
    N = 20
    x = da.random.default_rng().normal(size=(N, N, 100), chunks=(1, N, 100))
    with warnings.catch_warnings(record=True) as w:
        x = x.rechunk((N, 1, 100))
    assert not w