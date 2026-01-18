import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def test_sg_coeffs_deriv():
    i = np.array([-2.0, 0.0, 2.0, 4.0, 6.0])
    x = i ** 2 / 4
    dx = i / 2
    d2x = np.full_like(i, 0.5)
    for pos in range(x.size):
        coeffs0 = savgol_coeffs(5, 3, pos=pos, delta=2.0, use='dot')
        assert_allclose(coeffs0.dot(x), x[pos], atol=1e-10)
        coeffs1 = savgol_coeffs(5, 3, pos=pos, delta=2.0, use='dot', deriv=1)
        assert_allclose(coeffs1.dot(x), dx[pos], atol=1e-10)
        coeffs2 = savgol_coeffs(5, 3, pos=pos, delta=2.0, use='dot', deriv=2)
        assert_allclose(coeffs2.dot(x), d2x[pos], atol=1e-10)