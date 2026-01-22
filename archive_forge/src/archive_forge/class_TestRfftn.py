from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
class TestRfftn:
    dtype = None
    cdtype = None

    def setup_method(self):
        np.random.seed(1234)

    @pytest.mark.parametrize('dtype,cdtype,maxnlp', [(np.float64, np.complex128, 2000), (np.float32, np.complex64, 3500)])
    def test_definition(self, dtype, cdtype, maxnlp):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        y = rfftn(x)
        assert_equal(y.dtype, cdtype)
        assert_array_almost_equal_nulp(y, direct_rdftn(x), maxnlp)
        x = random((20, 26))
        assert_array_almost_equal_nulp(rfftn(x), direct_rdftn(x), maxnlp)
        x = random((5, 4, 3, 20))
        assert_array_almost_equal_nulp(rfftn(x), direct_rdftn(x), maxnlp)

    @pytest.mark.parametrize('size', [1, 2, 51, 32, 64, 92])
    def test_random(self, size):
        x = random([size, size])
        assert_allclose(irfftn(rfftn(x), x.shape), x, atol=1e-10)

    @pytest.mark.parametrize('func', [rfftn, irfftn])
    def test_invalid_sizes(self, func):
        with assert_raises(ValueError, match='invalid number of data points \\(\\[1, 0\\]\\) specified'):
            func([[]])
        with assert_raises(ValueError, match='invalid number of data points \\(\\[4, -3\\]\\) specified'):
            func([[1, 1], [2, 2]], (4, -3))

    @pytest.mark.parametrize('func', [rfftn, irfftn])
    def test_no_axes(self, func):
        with assert_raises(ValueError, match='at least 1 axis must be transformed'):
            func([], axes=[])

    def test_complex_input(self):
        with assert_raises(TypeError, match='x must be a real sequence'):
            rfftn(np.zeros(10, dtype=np.complex64))