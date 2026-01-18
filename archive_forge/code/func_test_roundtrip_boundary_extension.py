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
def test_roundtrip_boundary_extension(self):
    np.random.seed(1234)
    settings = [('boxcar', 100, 10, 0), ('boxcar', 100, 10, 9)]
    for window, N, nperseg, noverlap in settings:
        t = np.arange(N)
        x = 10 * np.random.randn(t.size)
        _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=True, boundary=None)
        _, xr = istft(zz, noverlap=noverlap, window=window, boundary=False)
        for boundary in ['even', 'odd', 'constant', 'zeros']:
            _, _, zz_ext = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=True, boundary=boundary)
            _, xr_ext = istft(zz_ext, noverlap=noverlap, window=window, boundary=True)
            msg = f'{window}, {noverlap}, {boundary}'
            assert_allclose(x, xr, err_msg=msg)
            assert_allclose(x, xr_ext, err_msg=msg)