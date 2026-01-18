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
def test_roundtrip_nola_not_cola(self):
    np.random.seed(1234)
    settings = [('boxcar', 100, 10, 3), ('bartlett', 101, 51, 37), ('hann', 1024, 256, 127), (('tukey', 0.5), 1152, 256, 14), ('hann', 1024, 256, 5)]
    for window, N, nperseg, noverlap in settings:
        msg = f'{window}, {nperseg}, {noverlap}'
        assert check_NOLA(window, nperseg, noverlap), msg
        assert not check_COLA(window, nperseg, noverlap), msg
        t = np.arange(N)
        x = 10 * np.random.randn(t.size)
        _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=True, boundary='zeros')
        tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap, window=window, boundary=True)
        msg = f'{window}, {noverlap}'
        assert_allclose(t, tr[:len(t)], err_msg=msg)
        assert_allclose(x, xr[:len(x)], err_msg=msg)