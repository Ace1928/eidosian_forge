import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_windowfunc_basics():
    for window_name, params in window_funcs:
        window = getattr(windows, window_name)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'This window is not suitable')
            w1 = window(8, *params, sym=True)
            w2 = window(7, *params, sym=False)
            assert_array_almost_equal(w1[:-1], w2)
            w1 = window(9, *params, sym=True)
            w2 = window(8, *params, sym=False)
            assert_array_almost_equal(w1[:-1], w2)
            assert_equal(len(window(6, *params, sym=True)), 6)
            assert_equal(len(window(6, *params, sym=False)), 6)
            assert_equal(len(window(7, *params, sym=True)), 7)
            assert_equal(len(window(7, *params, sym=False)), 7)
            assert_raises(ValueError, window, 5.5, *params)
            assert_raises(ValueError, window, -7, *params)
            assert_array_equal(window(0, *params, sym=True), [])
            assert_array_equal(window(0, *params, sym=False), [])
            assert_array_equal(window(1, *params, sym=True), [1])
            assert_array_equal(window(1, *params, sym=False), [1])
            assert_(window(0, *params, sym=True).dtype == 'float')
            assert_(window(0, *params, sym=False).dtype == 'float')
            assert_(window(1, *params, sym=True).dtype == 'float')
            assert_(window(1, *params, sym=False).dtype == 'float')
            assert_(window(6, *params, sym=True).dtype == 'float')
            assert_(window(6, *params, sym=False).dtype == 'float')
            assert_array_less(window(10, *params, sym=True), 1.01)
            assert_array_less(window(10, *params, sym=False), 1.01)
            assert_array_less(window(9, *params, sym=True), 1.01)
            assert_array_less(window(9, *params, sym=False), 1.01)
            assert_allclose(fft(window(10, *params, sym=False)).imag, 0, atol=1e-14)
            assert_allclose(fft(window(11, *params, sym=False)).imag, 0, atol=1e-14)