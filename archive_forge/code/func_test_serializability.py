from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
def test_serializability(generator_class):
    state = generator_class(5)
    x = state.normal(10, 1, size=10, chunks=5)
    y = _loads(_dumps(x))
    assert_eq(x, y)