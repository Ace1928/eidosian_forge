import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestBoxcar:

    def test_basic(self):
        assert_allclose(windows.boxcar(6), [1, 1, 1, 1, 1, 1])
        assert_allclose(windows.boxcar(7), [1, 1, 1, 1, 1, 1, 1])
        assert_allclose(windows.boxcar(6, False), [1, 1, 1, 1, 1, 1])