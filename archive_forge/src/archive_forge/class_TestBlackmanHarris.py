import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestBlackmanHarris:

    def test_basic(self):
        assert_allclose(windows.blackmanharris(6, False), [6e-05, 0.055645, 0.520575, 1.0, 0.520575, 0.055645])
        assert_allclose(windows.blackmanharris(7, sym=False), [6e-05, 0.03339172347815117, 0.332833504298565, 0.8893697722232837, 0.8893697722232838, 0.3328335042985652, 0.03339172347815122])
        assert_allclose(windows.blackmanharris(6), [6e-05, 0.1030114893456638, 0.7938335106543362, 0.7938335106543364, 0.1030114893456638, 6e-05])
        assert_allclose(windows.blackmanharris(7, sym=True), [6e-05, 0.055645, 0.520575, 1.0, 0.520575, 0.055645, 6e-05])