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
def test_nondefault_noverlap(self):
    x = np.zeros(64)
    x[::8] = 1
    f, p = csd(x, x, nperseg=16, noverlap=4)
    q = np.array([0, 1.0 / 12.0, 1.0 / 3.0, 1.0 / 5.0, 1.0 / 3.0, 1.0 / 5.0, 1.0 / 3.0, 1.0 / 5.0, 1.0 / 6.0])
    assert_allclose(p, q, atol=1e-12)