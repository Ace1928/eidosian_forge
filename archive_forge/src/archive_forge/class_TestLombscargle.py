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
class TestLombscargle:

    def test_frequency(self):
        """Test if frequency location of peak corresponds to frequency of
        generated input signal.
        """
        ampl = 2.0
        w = 1.0
        phi = 0.5 * np.pi
        nin = 100
        nout = 1000
        p = 0.7
        np.random.seed(2353425)
        r = np.random.rand(nin)
        t = np.linspace(0.01 * np.pi, 10.0 * np.pi, nin)[r >= p]
        x = ampl * np.sin(w * t + phi)
        f = np.linspace(0.01, 10.0, nout)
        P = lombscargle(t, x, f)
        delta = f[1] - f[0]
        assert_(w - f[np.argmax(P)] < delta / 2.0)

    def test_amplitude(self):
        ampl = 2.0
        w = 1.0
        phi = 0.5 * np.pi
        nin = 100
        nout = 1000
        p = 0.7
        np.random.seed(2353425)
        r = np.random.rand(nin)
        t = np.linspace(0.01 * np.pi, 10.0 * np.pi, nin)[r >= p]
        x = ampl * np.sin(w * t + phi)
        f = np.linspace(0.01, 10.0, nout)
        pgram = lombscargle(t, x, f)
        pgram = np.sqrt(4 * pgram / t.shape[0])
        assert_approx_equal(np.max(pgram), ampl, significant=2)

    def test_precenter(self):
        ampl = 2.0
        w = 1.0
        phi = 0.5 * np.pi
        nin = 100
        nout = 1000
        p = 0.7
        offset = 0.15
        np.random.seed(2353425)
        r = np.random.rand(nin)
        t = np.linspace(0.01 * np.pi, 10.0 * np.pi, nin)[r >= p]
        x = ampl * np.sin(w * t + phi) + offset
        f = np.linspace(0.01, 10.0, nout)
        pgram = lombscargle(t, x, f, precenter=True)
        pgram2 = lombscargle(t, x - x.mean(), f, precenter=False)
        assert_allclose(pgram, pgram2)

    def test_normalize(self):
        ampl = 2.0
        w = 1.0
        phi = 0.5 * np.pi
        nin = 100
        nout = 1000
        p = 0.7
        np.random.seed(2353425)
        r = np.random.rand(nin)
        t = np.linspace(0.01 * np.pi, 10.0 * np.pi, nin)[r >= p]
        x = ampl * np.sin(w * t + phi)
        f = np.linspace(0.01, 10.0, nout)
        pgram = lombscargle(t, x, f)
        pgram2 = lombscargle(t, x, f, normalize=True)
        assert_allclose(pgram * 2 / np.dot(x, x), pgram2)
        assert_approx_equal(np.max(pgram2), 1.0, significant=2)

    def test_wrong_shape(self):
        t = np.linspace(0, 1, 1)
        x = np.linspace(0, 1, 2)
        f = np.linspace(0, 1, 3)
        assert_raises(ValueError, lombscargle, t, x, f)

    def test_zero_division(self):
        t = np.zeros(1)
        x = np.zeros(1)
        f = np.zeros(1)
        assert_raises(ZeroDivisionError, lombscargle, t, x, f)

    def test_lombscargle_atan_vs_atan2(self):
        t = np.linspace(0, 10, 1000, endpoint=False)
        x = np.sin(4 * t)
        f = np.linspace(0, 50, 500, endpoint=False) + 0.1
        lombscargle(t, x, f * 2 * np.pi)