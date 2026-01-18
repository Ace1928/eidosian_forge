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
def test_resets():
    for klass, inp, outp in ((SlopeInterArrayWriter, (1, 511), (2.0, 1.0)), (SlopeArrayWriter, (0, 510), (2.0, 0.0))):
        arr = np.array(inp)
        outp = np.array(outp)
        aw = klass(arr, np.uint8)
        assert_array_equal(get_slope_inter(aw), outp)
        aw.calc_scale()
        assert_array_equal(get_slope_inter(aw), outp)
        aw.calc_scale(force=True)
        assert_array_equal(get_slope_inter(aw), outp)
        aw.array[:] = aw.array * 2
        aw.calc_scale()
        assert_array_equal(get_slope_inter(aw), outp)
        aw.calc_scale(force=True)
        assert_array_equal(get_slope_inter(aw), outp * 2)
        aw.reset()
        assert_array_equal(get_slope_inter(aw), (1.0, 0.0))