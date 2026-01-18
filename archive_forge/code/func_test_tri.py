from __future__ import annotations
import pytest
import numpy as np
import pytest
from tlz import concat
import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.array.numpy_compat import AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('N, M, k, dtype, chunks', [(3, None, 0, float, 'auto'), (4, None, 0, float, 'auto'), (3, 4, 0, bool, 'auto'), (3, None, 1, int, 'auto'), (3, None, -1, int, 'auto'), (3, None, 2, int, 1), (6, 8, -2, int, (3, 4)), (6, 8, 0, int, (3, 'auto'))])
def test_tri(N, M, k, dtype, chunks):
    assert_eq(da.tri(N, M, k, dtype, chunks), np.tri(N, M, k, dtype))