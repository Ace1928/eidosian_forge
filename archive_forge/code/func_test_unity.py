import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_unity(self):
    for M in range(1, 21):
        win = windows.dpss(M, M / 2.1)
        expected = M % 2
        assert_equal(np.isclose(win, 1.0).sum(), expected, err_msg=f'{win}')
        win_sub = windows.dpss(M, M / 2.1, norm='subsample')
        if M > 2:
            assert_equal(np.isclose(win_sub, 1.0).sum(), expected, err_msg=f'{win_sub}')
            assert_allclose(win, win_sub, rtol=0.03)
        win_2 = windows.dpss(M, M / 2.1, norm=2)
        expected = 1 if M == 1 else 0
        assert_equal(np.isclose(win_2, 1.0).sum(), expected, err_msg=f'{win_2}')