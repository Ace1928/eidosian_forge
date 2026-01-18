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
@pytest.mark.parametrize('scaling', ['spectrum', 'psd'])
def test_roundtrip_real(self, scaling):
    np.random.seed(1234)
    settings = [('boxcar', 100, 10, 0), ('boxcar', 100, 10, 9), ('bartlett', 101, 51, 26), ('hann', 1024, 256, 128), (('tukey', 0.5), 1152, 256, 64), ('hann', 1024, 256, 255)]
    for window, N, nperseg, noverlap in settings:
        t = np.arange(N)
        x = 10 * np.random.randn(t.size)
        _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=False, scaling=scaling)
        tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap, window=window, scaling=scaling)
        msg = f'{window}, {noverlap}'
        assert_allclose(t, tr, err_msg=msg)
        assert_allclose(x, xr, err_msg=msg)