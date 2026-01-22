import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestReal:

    def test_real(self):
        y = np.random.rand(10)
        assert_array_equal(y, np.real(y))
        y = np.array(1)
        out = np.real(y)
        assert_array_equal(y, out)
        assert_(isinstance(out, np.ndarray))
        y = 1
        out = np.real(y)
        assert_equal(y, out)
        assert_(not isinstance(out, np.ndarray))

    def test_cmplx(self):
        y = np.random.rand(10) + 1j * np.random.rand(10)
        assert_array_equal(y.real, np.real(y))
        y = np.array(1 + 1j)
        out = np.real(y)
        assert_array_equal(y.real, out)
        assert_(isinstance(out, np.ndarray))
        y = 1 + 1j
        out = np.real(y)
        assert_equal(1.0, out)
        assert_(not isinstance(out, np.ndarray))