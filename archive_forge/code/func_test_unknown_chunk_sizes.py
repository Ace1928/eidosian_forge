from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask.array.utils import assert_eq, same_keys
@percentile_internal_methods
def test_unknown_chunk_sizes(internal_method):
    x = da.random.default_rng().random(1000, chunks=(100,))
    x._chunks = ((np.nan,) * 10,)
    result = da.percentile(x, 50, internal_method=internal_method).compute()
    assert 0.1 < result < 0.9
    a, b = da.percentile(x, [40, 60], internal_method=internal_method).compute()
    assert 0.1 < a < 0.9
    assert 0.1 < b < 0.9
    assert a < b