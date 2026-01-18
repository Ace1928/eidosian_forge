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
def test_padded_freqs(self):
    x = np.zeros(12)
    y = np.ones(12)
    nfft = 24
    f = fftfreq(nfft, 1.0)[:nfft // 2 + 1]
    f[-1] *= -1
    fodd, _ = csd(x, y, nperseg=5, nfft=nfft)
    feven, _ = csd(x, y, nperseg=6, nfft=nfft)
    assert_allclose(f, fodd)
    assert_allclose(f, feven)
    nfft = 25
    f = fftfreq(nfft, 1.0)[:(nfft + 1) // 2]
    fodd, _ = csd(x, y, nperseg=5, nfft=nfft)
    feven, _ = csd(x, y, nperseg=6, nfft=nfft)
    assert_allclose(f, fodd)
    assert_allclose(f, feven)