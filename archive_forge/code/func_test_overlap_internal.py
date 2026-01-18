from __future__ import annotations
import numpy as np
import pytest
from packaging.version import parse as parse_version
import dask.array as da
from dask.array.utils import assert_eq, same_keys
def test_overlap_internal():
    x = cupy.arange(64).reshape((8, 8))
    d = da.from_array(x, chunks=(4, 4), asarray=False)
    g = da.overlap.overlap_internal(d, {0: 2, 1: 1})
    assert g.chunks == ((6, 6), (5, 5))
    expected = np.array([[0, 1, 2, 3, 4, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 27, 28, 29, 30, 31], [32, 33, 34, 35, 36, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 43, 44, 45, 46, 47], [16, 17, 18, 19, 20, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 27, 28, 29, 30, 31], [32, 33, 34, 35, 36, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 43, 44, 45, 46, 47], [48, 49, 50, 51, 52, 51, 52, 53, 54, 55], [56, 57, 58, 59, 60, 59, 60, 61, 62, 63]])
    assert_eq(g, expected, check_type=False)
    assert same_keys(da.overlap.overlap_internal(d, {0: 2, 1: 1}), g)