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
def test_rechunk_intermediates():
    x = da.random.default_rng().normal(10, 0.1, (10, 10), chunks=(10, 1))
    y = x.rechunk((1, 10))
    assert len(y.dask) > 30