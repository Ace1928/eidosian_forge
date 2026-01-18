import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def test_sg_coeffs_exact():
    polyorder = 4
    window_length = 9
    halflen = window_length // 2
    x = np.linspace(0, 21, 43)
    delta = x[1] - x[0]
    y = 0.5 * x ** 3 - x
    h = savgol_coeffs(window_length, polyorder)
    y0 = convolve1d(y, h)
    assert_allclose(y0[halflen:-halflen], y[halflen:-halflen])
    dy = 1.5 * x ** 2 - 1
    h = savgol_coeffs(window_length, polyorder, deriv=1, delta=delta)
    y1 = convolve1d(y, h)
    assert_allclose(y1[halflen:-halflen], dy[halflen:-halflen])
    d2y = 3.0 * x
    h = savgol_coeffs(window_length, polyorder, deriv=2, delta=delta)
    y2 = convolve1d(y, h)
    assert_allclose(y2[halflen:-halflen], d2y[halflen:-halflen])