from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def test_shape_axes_argument2(self):
    x = numpy.random.random((10, 5, 3, 7))
    y = fftn(x, axes=(-1,), s=(8,))
    assert_array_almost_equal(y, fft(x, axis=-1, n=8))
    x = numpy.random.random((10, 5, 3, 7))
    y = fftn(x, axes=(-2,), s=(8,))
    assert_array_almost_equal(y, fft(x, axis=-2, n=8))
    x = numpy.random.random((4, 4, 2))
    y = fftn(x, axes=(-3, -2), s=(8, 8))
    assert_array_almost_equal(y, numpy.fft.fftn(x, axes=(-3, -2), s=(8, 8)))