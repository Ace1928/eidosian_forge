from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def test_regression_244(self):
    """FFT returns wrong result with axes parameter."""
    x = numpy.ones((4, 4, 2))
    y = fftn(x, s=(8, 8), axes=(-3, -2))
    y_r = numpy.fft.fftn(x, s=(8, 8), axes=(-3, -2))
    assert_allclose(y, y_r)