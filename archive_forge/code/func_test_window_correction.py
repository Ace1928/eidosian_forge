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
def test_window_correction(self):
    A = 20
    fs = 10000.0
    nperseg = int(fs // 10)
    fsig = 300
    ii = int(fsig * nperseg // fs)
    tt = np.arange(fs) / fs
    x = A * np.sin(2 * np.pi * fsig * tt)
    for window in ['hann', 'bartlett', ('tukey', 0.1), 'flattop']:
        _, p_spec = welch(x, fs=fs, nperseg=nperseg, window=window, scaling='spectrum')
        freq, p_dens = welch(x, fs=fs, nperseg=nperseg, window=window, scaling='density')
        assert_allclose(p_spec[ii], A ** 2 / 2.0)
        assert_allclose(np.sqrt(trapezoid(p_dens, freq)), A * np.sqrt(2) / 2, rtol=0.001)