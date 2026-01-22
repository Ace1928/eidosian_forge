from . import util
import numpy as np
import pytest
from numpy.testing import assert_allclose
class TestISOC(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'isocintrin', 'isoCtests.f90')]

    def test_c_double(self):
        out = self.module.coddity.c_add(1, 2)
        exp_out = 3
        assert out == exp_out

    def test_bindc_function(self):
        out = self.module.coddity.wat(1, 20)
        exp_out = 8
        assert out == exp_out

    def test_bindc_kinds(self):
        out = self.module.coddity.c_add_int64(1, 20)
        exp_out = 21
        assert out == exp_out

    def test_bindc_add_arr(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        out = self.module.coddity.add_arr(a, b)
        exp_out = a * 2
        assert_allclose(out, exp_out)