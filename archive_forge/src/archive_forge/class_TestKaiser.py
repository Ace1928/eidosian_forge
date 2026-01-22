import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestKaiser:

    def test_basic(self):
        assert_allclose(windows.kaiser(6, 0.5), [0.9403061933191572, 0.9782962393705389, 0.9975765035372042, 0.9975765035372042, 0.9782962393705389, 0.9403061933191572])
        assert_allclose(windows.kaiser(7, 0.5), [0.9403061933191572, 0.9732402256999829, 0.9932754654413773, 1.0, 0.9932754654413773, 0.9732402256999829, 0.9403061933191572])
        assert_allclose(windows.kaiser(6, 2.7), [0.2603047507678832, 0.6648106293528054, 0.9582099802511439, 0.9582099802511439, 0.6648106293528054, 0.2603047507678832])
        assert_allclose(windows.kaiser(7, 2.7), [0.2603047507678832, 0.5985765418119844, 0.8868495172060835, 1.0, 0.8868495172060835, 0.5985765418119844, 0.2603047507678832])
        assert_allclose(windows.kaiser(6, 2.7, False), [0.2603047507678832, 0.5985765418119844, 0.8868495172060835, 1.0, 0.8868495172060835, 0.5985765418119844])