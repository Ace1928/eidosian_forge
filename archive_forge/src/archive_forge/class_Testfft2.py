from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fftpack import ifft, fft, fftn, ifftn, rfft, irfft, fft2
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
class Testfft2:

    def setup_method(self):
        np.random.seed(1234)

    def test_regression_244(self):
        """FFT returns wrong result with axes parameter."""
        x = numpy.ones((4, 4, 2))
        y = fft2(x, shape=(8, 8), axes=(-3, -2))
        y_r = numpy.fft.fftn(x, s=(8, 8), axes=(-3, -2))
        assert_array_almost_equal(y, y_r)

    def test_invalid_sizes(self):
        assert_raises(ValueError, fft2, [[]])
        assert_raises(ValueError, fft2, [[1, 1], [2, 2]], (4, -3))