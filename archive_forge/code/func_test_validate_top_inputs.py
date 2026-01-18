from __future__ import annotations
import collections
from operator import add
import numpy as np
import pytest
import dask
import dask.array as da
from dask.array.utils import assert_eq
from dask.blockwise import (
from dask.highlevelgraph import HighLevelGraph
from dask.utils_test import dec, hlg_layer_topological, inc
def test_validate_top_inputs():
    A = da.random.default_rng().random((20, 20), chunks=(10, 10))
    with pytest.raises(ValueError) as info:
        da.blockwise(inc, 'jk', A, 'ij', dtype=A.dtype)
    assert 'unknown dimension' in str(info.value).lower()
    assert 'k' in str(info.value)
    assert 'j' not in str(info.value)
    with pytest.raises(ValueError) as info:
        da.blockwise(inc, 'ii', A, 'ij', dtype=A.dtype)
    assert 'repeated' in str(info.value).lower()
    assert 'i' in str(info.value)