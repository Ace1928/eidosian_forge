from __future__ import annotations
import itertools
import warnings
import pytest
from tlz import merge
import dask
import dask.array as da
from dask import config
from dask.array.chunk import getitem
from dask.array.slicing import (
from dask.array.utils import assert_eq, same_keys
def test_normalize_index():
    assert normalize_index((Ellipsis, None), (10,)) == (slice(None), None)
    assert normalize_index(5, (np.nan,)) == (5,)
    assert normalize_index(-5, (np.nan,)) == (-5,)
    result, = normalize_index([-5, -2, 1], (np.nan,))
    assert result.tolist() == [-5, -2, 1]
    assert normalize_index(slice(-5, -2), (np.nan,)) == (slice(-5, -2),)