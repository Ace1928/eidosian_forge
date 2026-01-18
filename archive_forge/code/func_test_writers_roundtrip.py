import itertools
from io import BytesIO
from platform import machine, python_compiler
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..arraywriters import (
from ..casting import int_abs, sctypes, shared_range, type_info
from ..testing import assert_allclose_safely, suppress_warnings
from ..volumeutils import _dt_min_max, apply_read_scaling, array_from_file
def test_writers_roundtrip():
    ndt = np.dtype(np.float64)
    arr = np.arange(3, dtype=ndt)
    aw = SlopeInterArrayWriter(arr, ndt, calc_scale=False)
    aw.inter = 1.0
    data_back = round_trip(aw)
    assert_array_equal(data_back, arr)
    aw.slope = 2.0
    data_back = round_trip(aw)
    assert_array_equal(data_back, arr)
    aw = SlopeInterArrayWriter(arr + np.nan, np.int32)
    data_back = round_trip(aw)
    assert_array_equal(data_back, np.zeros(arr.shape))
    arr[0] = np.inf
    aw = SlopeInterArrayWriter(arr, np.int32)
    data_back = round_trip(aw)
    assert_array_almost_equal(data_back, [2, 1, 2])