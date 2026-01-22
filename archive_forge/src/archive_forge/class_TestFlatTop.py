import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestFlatTop:

    def test_basic(self):
        assert_allclose(windows.flattop(6, sym=False), [-0.000421051, -0.051263156, 0.19821053, 1.0, 0.19821053, -0.051263156])
        assert_allclose(windows.flattop(7, sym=False), [-0.000421051, -0.03684078115492348, 0.01070371671615342, 0.7808739149387698, 0.7808739149387698, 0.01070371671615342, -0.03684078115492348])
        assert_allclose(windows.flattop(6), [-0.000421051, -0.0677142520762119, 0.6068721525762117, 0.6068721525762117, -0.0677142520762119, -0.000421051])
        assert_allclose(windows.flattop(7, True), [-0.000421051, -0.051263156, 0.19821053, 1.0, 0.19821053, -0.051263156, -0.000421051])