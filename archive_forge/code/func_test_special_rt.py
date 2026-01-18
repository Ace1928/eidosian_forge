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
def test_special_rt():
    arr = np.array([np.inf, np.nan, -np.inf])
    for in_dtt in FLOAT_TYPES:
        for out_dtt in IUINT_TYPES:
            in_arr = arr.astype(in_dtt)
            with pytest.raises(WriterError):
                ArrayWriter(in_arr, out_dtt)
            aw = ArrayWriter(in_arr, out_dtt, check_scaling=False)
            mn, mx = shared_range(float, out_dtt)
            assert np.allclose(round_trip(aw).astype(float), [mx, 0, mn])
            for klass in (SlopeArrayWriter, SlopeInterArrayWriter):
                aw = klass(in_arr, out_dtt)
                assert get_slope_inter(aw) == (1, 0)
                assert_array_equal(round_trip(aw), 0)
    for in_dtt, out_dtt, awt in itertools.product(FLOAT_TYPES, IUINT_TYPES, (ArrayWriter, SlopeArrayWriter, SlopeInterArrayWriter)):
        arr = np.zeros((3,), dtype=in_dtt)
        aw = awt(arr, out_dtt)
        assert get_slope_inter(aw) == (1, 0)
        assert_array_equal(round_trip(aw), 0)