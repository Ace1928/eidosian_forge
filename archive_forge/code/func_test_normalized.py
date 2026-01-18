import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_normalized(self):
    """Tests windows of small length that are normalized to 1. See the
        documentation for the Taylor window for more information on
        normalization.
        """
    assert_allclose(windows.taylor(1, 2, 15), 1.0)
    assert_allclose(windows.taylor(5, 2, 15), np.array([0.75803341, 0.90757699, 1.0, 0.90757699, 0.75803341]))
    assert_allclose(windows.taylor(6, 2, 15), np.array([0.7504082, 0.86624416, 0.98208011, 0.98208011, 0.86624416, 0.7504082]))