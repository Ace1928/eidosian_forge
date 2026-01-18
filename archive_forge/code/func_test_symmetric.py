import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_symmetric():
    for win in [windows.lanczos]:
        w = win(4096)
        error = np.max(np.abs(w - np.flip(w)))
        assert_equal(error, 0.0)
        w = win(4097)
        error = np.max(np.abs(w - np.flip(w)))
        assert_equal(error, 0.0)