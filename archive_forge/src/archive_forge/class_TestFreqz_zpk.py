import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestFreqz_zpk:

    def test_ticket1441(self):
        """Regression test for ticket 1441."""
        N = 100000
        w, h = freqz_zpk([0.5], [0.5], 1.0, worN=N)
        assert_equal(w.shape, (N,))

    def test_basic(self):
        w, h = freqz_zpk([0.5], [0.5], 1.0, worN=8)
        assert_array_almost_equal(w, np.pi * np.arange(8.0) / 8)
        assert_array_almost_equal(h, np.ones(8))

    def test_basic_whole(self):
        w, h = freqz_zpk([0.5], [0.5], 1.0, worN=8, whole=True)
        assert_array_almost_equal(w, 2 * np.pi * np.arange(8.0) / 8)
        assert_array_almost_equal(h, np.ones(8))

    def test_vs_freqz(self):
        b, a = cheby1(4, 5, 0.5, analog=False, output='ba')
        z, p, k = cheby1(4, 5, 0.5, analog=False, output='zpk')
        w1, h1 = freqz(b, a)
        w2, h2 = freqz_zpk(z, p, k)
        assert_allclose(w1, w2)
        assert_allclose(h1, h2, rtol=1e-06)

    def test_backward_compat(self):
        w1, h1 = freqz_zpk([0.5], [0.5], 1.0)
        w2, h2 = freqz_zpk([0.5], [0.5], 1.0, None)
        assert_array_almost_equal(w1, w2)
        assert_array_almost_equal(h1, h2)

    def test_fs_param(self):
        fs = 900
        z = [-1, -1, -1]
        p = [0.4747869998473389 + 0.4752230717749344j, 0.37256600288916636, 0.4747869998473389 - 0.4752230717749344j]
        k = 0.03934683014103762
        w1, h1 = freqz_zpk(z, p, k, whole=False, fs=fs)
        w2, h2 = freqz_zpk(z, p, k, whole=False)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs / 2, 512, endpoint=False))
        w1, h1 = freqz_zpk(z, p, k, whole=True, fs=fs)
        w2, h2 = freqz_zpk(z, p, k, whole=True)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs, 512, endpoint=False))
        w1, h1 = freqz_zpk(z, p, k, 5, fs=fs)
        w2, h2 = freqz_zpk(z, p, k, 5)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs / 2, 5, endpoint=False))
        w1, h1 = freqz_zpk(z, p, k, 5, whole=True, fs=fs)
        w2, h2 = freqz_zpk(z, p, k, 5, whole=True)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs, 5, endpoint=False))
        for w in ([123], (123,), np.array([123]), (50, 123, 230), np.array([50, 123, 230])):
            w1, h1 = freqz_zpk(z, p, k, w, fs=fs)
            w2, h2 = freqz_zpk(z, p, k, 2 * pi * np.array(w) / fs)
            assert_allclose(h1, h2)
            assert_allclose(w, w1)

    def test_w_or_N_types(self):
        for N in (8, np.int8(8), np.int16(8), np.int32(8), np.int64(8), np.array(8)):
            w, h = freqz_zpk([], [], 1, worN=N)
            assert_array_almost_equal(w, np.pi * np.arange(8) / 8.0)
            assert_array_almost_equal(h, np.ones(8))
            w, h = freqz_zpk([], [], 1, worN=N, fs=100)
            assert_array_almost_equal(w, np.linspace(0, 50, 8, endpoint=False))
            assert_array_almost_equal(h, np.ones(8))
        for w in (8.0, 8.0 + 0j):
            w_out, h = freqz_zpk([], [], 1, worN=w, fs=100)
            assert_array_almost_equal(w_out, [8])
            assert_array_almost_equal(h, [1])