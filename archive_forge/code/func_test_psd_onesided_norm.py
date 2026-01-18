from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_psd_onesided_norm():
    u = np.array([0, 1, 2, 3, 1, 2, 1])
    dt = 1.0
    Su = np.abs(np.fft.fft(u) * dt) ** 2 / (dt * u.size)
    P, f = mlab.psd(u, NFFT=u.size, Fs=1 / dt, window=mlab.window_none, detrend=mlab.detrend_none, noverlap=0, pad_to=None, scale_by_freq=None, sides='onesided')
    Su_1side = np.append([Su[0]], Su[1:4] + Su[4:][::-1])
    assert_allclose(P, Su_1side, atol=1e-06)