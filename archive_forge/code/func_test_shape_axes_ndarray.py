from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fftpack import ifft, fft, fftn, ifftn, rfft, irfft, fft2
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
@pytest.mark.parametrize('func', [fftn, ifftn, fft2])
def test_shape_axes_ndarray(func):
    a = np.random.rand(10, 10)
    expect = func(a, shape=(5, 5))
    actual = func(a, shape=np.array([5, 5]))
    assert_equal(expect, actual)
    expect = func(a, axes=(-1,))
    actual = func(a, axes=np.array([-1]))
    assert_equal(expect, actual)
    expect = func(a, shape=(4, 7), axes=(1, 0))
    actual = func(a, shape=np.array([4, 7]), axes=np.array([1, 0]))
    assert_equal(expect, actual)