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
def test_padded_fft(self):
    x = np.zeros(16)
    x[0] = 1
    f, p = periodogram(x)
    fp, pp = periodogram(x, nfft=32)
    assert_allclose(f, fp[::2])
    assert_allclose(p, pp[::2])
    assert_array_equal(pp.shape, (17,))