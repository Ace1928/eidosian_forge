import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestImag:

    def test_real(self):
        y = np.random.rand(10)
        assert_array_equal(0, np.imag(y))
        y = np.array(1)
        out = np.imag(y)
        assert_array_equal(0, out)
        assert_(isinstance(out, np.ndarray))
        y = 1
        out = np.imag(y)
        assert_equal(0, out)
        assert_(not isinstance(out, np.ndarray))

    def test_cmplx(self):
        y = np.random.rand(10) + 1j * np.random.rand(10)
        assert_array_equal(y.imag, np.imag(y))
        y = np.array(1 + 1j)
        out = np.imag(y)
        assert_array_equal(y.imag, out)
        assert_(isinstance(out, np.ndarray))
        y = 1 + 1j
        out = np.imag(y)
        assert_equal(1.0, out)
        assert_(not isinstance(out, np.ndarray))