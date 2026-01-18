from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.utils import assert_eq, same_keys
def test_percentiles_with_unknown_chunk_sizes():
    rng = da.random.default_rng(cupy.random.default_rng())
    x = rng.random(1000, chunks=(100,))
    x._chunks = ((np.nan,) * 10,)
    result = da.percentile(x, 50, method='midpoint').compute()
    assert type(result) == cupy.ndarray
    assert 0.1 < result < 0.9
    a, b = da.percentile(x, [40, 60], method='midpoint').compute()
    assert type(a) == cupy.ndarray
    assert type(b) == cupy.ndarray
    assert 0.1 < a < 0.9
    assert 0.1 < b < 0.9
    assert a < b