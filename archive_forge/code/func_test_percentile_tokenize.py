from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.utils import assert_eq, same_keys
def test_percentile_tokenize():
    d = da.from_array(cupy.ones((16,)), chunks=(4,))
    qs = np.array([0, 50, 100])
    assert same_keys(da.percentile(d, qs), da.percentile(d, qs))