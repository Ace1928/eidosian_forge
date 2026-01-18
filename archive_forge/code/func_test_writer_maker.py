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
def test_writer_maker():
    arr = np.arange(10, dtype=np.float64)
    aw = make_array_writer(arr, np.float64)
    assert isinstance(aw, SlopeInterArrayWriter)
    aw = make_array_writer(arr, np.float64, True, True)
    assert isinstance(aw, SlopeInterArrayWriter)
    aw = make_array_writer(arr, np.float64, True, False)
    assert isinstance(aw, SlopeArrayWriter)
    aw = make_array_writer(arr, np.float64, False, False)
    assert isinstance(aw, ArrayWriter)
    with pytest.raises(ValueError):
        make_array_writer(arr, np.float64, False)
    with pytest.raises(ValueError):
        make_array_writer(arr, np.float64, False, True)
    aw = make_array_writer(arr, np.int16, calc_scale=False)
    assert (aw.slope, aw.inter) == (1, 0)
    aw.calc_scale()
    slope, inter = (aw.slope, aw.inter)
    assert not (slope, inter) == (1, 0)
    aw = make_array_writer(arr, np.int16)
    assert (aw.slope, aw.inter) == (slope, inter)
    aw = make_array_writer(arr, np.int16, calc_scale=True)
    assert (aw.slope, aw.inter) == (slope, inter)