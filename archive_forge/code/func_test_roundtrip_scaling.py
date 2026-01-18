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
def test_roundtrip_scaling(self):
    """Verify behavior of scaling parameter. """
    X = np.zeros(513, dtype=complex)
    X[256] = 1024
    x = np.fft.irfft(X)
    power_x = sum(x ** 2) / len(x)
    Zs = stft(x, boundary='even', scaling='spectrum')[2]
    x1 = istft(Zs, boundary=True, scaling='spectrum')[1]
    assert_allclose(x1, x)
    assert_allclose(abs(Zs[63, :-1]), 0.5)
    assert_allclose(abs(Zs[64, :-1]), 1)
    assert_allclose(abs(Zs[65, :-1]), 0.5)
    Zs[63:66, :-1] = 0
    assert_allclose(Zs[:, :-1], 0, atol=np.finfo(Zs.dtype).resolution)
    Zp = stft(x, return_onesided=False, boundary='even', scaling='psd')[2]
    psd_Zp = np.sum(Zp.real ** 2 + Zp.imag ** 2, axis=0) / Zp.shape[0]
    assert_allclose(psd_Zp, power_x)
    x1 = istft(Zp, input_onesided=False, boundary=True, scaling='psd')[1]
    assert_allclose(x1, x)
    Zp0 = stft(x, return_onesided=True, boundary='even', scaling='psd')[2]
    Zp1 = np.conj(Zp0[-2:0:-1, :])
    assert_allclose(Zp[:129, :], Zp0)
    assert_allclose(Zp[129:, :], Zp1)
    s2 = np.sum(Zp0.real ** 2 + Zp0.imag ** 2, axis=0) + np.sum(Zp1.real ** 2 + Zp1.imag ** 2, axis=0)
    psd_Zp01 = s2 / (Zp0.shape[0] + Zp1.shape[0])
    assert_allclose(psd_Zp01, power_x)
    x1 = istft(Zp0, input_onesided=True, boundary=True, scaling='psd')[1]
    assert_allclose(x1, x)