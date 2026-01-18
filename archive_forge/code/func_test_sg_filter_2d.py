import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def test_sg_filter_2d():
    x = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0]])
    expected = np.array([[1.0, 4.0 / 3, 1.0], [2.0, 8.0 / 3, 2.0]])
    y = savgol_filter(x, 3, 1, mode='constant')
    assert_allclose(y, expected)
    y = savgol_filter(x.T, 3, 1, mode='constant', axis=0)
    assert_allclose(y, expected.T)