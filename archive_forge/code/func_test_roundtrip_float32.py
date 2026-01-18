import sys
import numpy as np
from numpy.testing import (assert_, assert_approx_equal,
import pytest
from pytest import raises as assert_raises
from scipy import signal
from scipy.fft import fftfreq
from scipy.integrate import trapezoid
from scipy.signal import (periodogram, welch, lombscargle, coherence,
from scipy.signal._spectral_py import _spectral_helper
from scipy.signal.tests._scipy_spectral_test_shim import stft_compare as stft
from scipy.signal.tests._scipy_spectral_test_shim import istft_compare as istft
from scipy.signal.tests._scipy_spectral_test_shim import csd_compare as csd
def test_roundtrip_float32(self):
    np.random.seed(1234)
    settings = [('hann', 1024, 256, 128)]
    for window, N, nperseg, noverlap in settings:
        t = np.arange(N)
        x = 10 * np.random.randn(t.size)
        x = x.astype(np.float32)
        _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=False)
        tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap, window=window)
        msg = f'{window}, {noverlap}'
        assert_allclose(t, t, err_msg=msg)
        assert_allclose(x, xr, err_msg=msg, rtol=0.0001, atol=1e-05)
        assert_(x.dtype == xr.dtype)