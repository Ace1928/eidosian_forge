import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
def alt_sg_coeffs(window_length, polyorder, pos):
    """This is an alternative implementation of the SG coefficients.

    It uses numpy.polyfit and numpy.polyval. The results should be
    equivalent to those of savgol_coeffs(), but this implementation
    is slower.

    window_length should be odd.

    """
    if pos is None:
        pos = window_length // 2
    t = np.arange(window_length)
    unit = (t == pos).astype(int)
    h = np.polyval(np.polyfit(t, unit, polyorder), t)
    return h