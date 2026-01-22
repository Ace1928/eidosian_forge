import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestBlackman:

    def test_basic(self):
        assert_allclose(windows.blackman(6, sym=False), [0, 0.13, 0.63, 1.0, 0.63, 0.13], atol=1e-14)
        assert_allclose(windows.blackman(7, sym=False), [0, 0.09045342435412804, 0.4591829575459636, 0.9203636180999081, 0.9203636180999081, 0.4591829575459636, 0.09045342435412804], atol=1e-08)
        assert_allclose(windows.blackman(6), [0, 0.2007701432625305, 0.8492298567374694, 0.8492298567374694, 0.2007701432625305, 0], atol=1e-14)
        assert_allclose(windows.blackman(7, True), [0, 0.13, 0.63, 1.0, 0.63, 0.13, 0], atol=1e-14)