import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestFreqz:

    def test_ticket1441(self):
        """Regression test for ticket 1441."""
        N = 100000
        w, h = freqz([1.0], worN=N)
        assert_equal(w.shape, (N,))

    def test_basic(self):
        w, h = freqz([1.0], worN=8)
        assert_array_almost_equal(w, np.pi * np.arange(8) / 8.0)
        assert_array_almost_equal(h, np.ones(8))
        w, h = freqz([1.0], worN=9)
        assert_array_almost_equal(w, np.pi * np.arange(9) / 9.0)
        assert_array_almost_equal(h, np.ones(9))
        for a in [1, np.ones(2)]:
            w, h = freqz(np.ones(2), a, worN=0)
            assert_equal(w.shape, (0,))
            assert_equal(h.shape, (0,))
            assert_equal(h.dtype, np.dtype('complex128'))
        t = np.linspace(0, 1, 4, endpoint=False)
        for b, a, h_whole in zip(([1.0, 0, 0, 0], np.sin(2 * np.pi * t)), ([1.0, 0, 0, 0], [0.5, 0, 0, 0]), ([1.0, 1.0, 1.0, 1.0], [0, -4j, 0, 4j])):
            w, h = freqz(b, a, worN=4, whole=True)
            expected_w = np.linspace(0, 2 * np.pi, 4, endpoint=False)
            assert_array_almost_equal(w, expected_w)
            assert_array_almost_equal(h, h_whole)
            w, h = freqz(b, a, worN=np.int32(4), whole=True)
            assert_array_almost_equal(w, expected_w)
            assert_array_almost_equal(h, h_whole)
            w, h = freqz(b, a, worN=w, whole=True)
            assert_array_almost_equal(w, expected_w)
            assert_array_almost_equal(h, h_whole)

    def test_basic_whole(self):
        w, h = freqz([1.0], worN=8, whole=True)
        assert_array_almost_equal(w, 2 * np.pi * np.arange(8.0) / 8)
        assert_array_almost_equal(h, np.ones(8))

    def test_plot(self):

        def plot(w, h):
            assert_array_almost_equal(w, np.pi * np.arange(8.0) / 8)
            assert_array_almost_equal(h, np.ones(8))
        assert_raises(ZeroDivisionError, freqz, [1.0], worN=8, plot=lambda w, h: 1 / 0)
        freqz([1.0], worN=8, plot=plot)

    def test_fft_wrapping(self):
        bs = list()
        as_ = list()
        hs_whole = list()
        hs_half = list()
        t = np.linspace(0, 1, 3, endpoint=False)
        bs.append(np.sin(2 * np.pi * t))
        as_.append(3.0)
        hs_whole.append([0, -0.5j, 0.5j])
        hs_half.append([0, np.sqrt(1.0 / 12.0), -0.5j])
        t = np.linspace(0, 1, 4, endpoint=False)
        bs.append(np.sin(2 * np.pi * t))
        as_.append(0.5)
        hs_whole.append([0, -4j, 0, 4j])
        hs_half.append([0, np.sqrt(8), -4j, -np.sqrt(8)])
        del t
        for ii, b in enumerate(bs):
            a = as_[ii]
            expected_w = np.linspace(0, 2 * np.pi, len(b), endpoint=False)
            w, h = freqz(b, a, worN=expected_w, whole=True)
            err_msg = f'b = {b}, a={a}'
            assert_array_almost_equal(w, expected_w, err_msg=err_msg)
            assert_array_almost_equal(h, hs_whole[ii], err_msg=err_msg)
            w, h = freqz(b, a, worN=len(b), whole=True)
            assert_array_almost_equal(w, expected_w, err_msg=err_msg)
            assert_array_almost_equal(h, hs_whole[ii], err_msg=err_msg)
            expected_w = np.linspace(0, np.pi, len(b), endpoint=False)
            w, h = freqz(b, a, worN=expected_w, whole=False)
            assert_array_almost_equal(w, expected_w, err_msg=err_msg)
            assert_array_almost_equal(h, hs_half[ii], err_msg=err_msg)
            w, h = freqz(b, a, worN=len(b), whole=False)
            assert_array_almost_equal(w, expected_w, err_msg=err_msg)
            assert_array_almost_equal(h, hs_half[ii], err_msg=err_msg)
        rng = np.random.RandomState(0)
        for ii in range(2, 10):
            b = rng.randn(ii)
            for kk in range(2):
                a = rng.randn(1) if kk == 0 else rng.randn(3)
                for jj in range(2):
                    if jj == 1:
                        b = b + rng.randn(ii) * 1j
                    expected_w = np.linspace(0, 2 * np.pi, ii, endpoint=False)
                    w, expected_h = freqz(b, a, worN=expected_w, whole=True)
                    assert_array_almost_equal(w, expected_w)
                    w, h = freqz(b, a, worN=ii, whole=True)
                    assert_array_almost_equal(w, expected_w)
                    assert_array_almost_equal(h, expected_h)
                    expected_w = np.linspace(0, np.pi, ii, endpoint=False)
                    w, expected_h = freqz(b, a, worN=expected_w, whole=False)
                    assert_array_almost_equal(w, expected_w)
                    w, h = freqz(b, a, worN=ii, whole=False)
                    assert_array_almost_equal(w, expected_w)
                    assert_array_almost_equal(h, expected_h)

    def test_broadcasting1(self):
        np.random.seed(123)
        b = np.random.rand(3, 5, 1)
        a = np.random.rand(2, 1)
        for whole in [False, True]:
            for worN in [16, 17, np.linspace(0, 1, 10), np.array([])]:
                w, h = freqz(b, a, worN=worN, whole=whole)
                for k in range(b.shape[1]):
                    bk = b[:, k, 0]
                    ak = a[:, 0]
                    ww, hh = freqz(bk, ak, worN=worN, whole=whole)
                    assert_allclose(ww, w)
                    assert_allclose(hh, h[k])

    def test_broadcasting2(self):
        np.random.seed(123)
        b = np.random.rand(3, 5, 1)
        for whole in [False, True]:
            for worN in [16, 17, np.linspace(0, 1, 10)]:
                w, h = freqz(b, worN=worN, whole=whole)
                for k in range(b.shape[1]):
                    bk = b[:, k, 0]
                    ww, hh = freqz(bk, worN=worN, whole=whole)
                    assert_allclose(ww, w)
                    assert_allclose(hh, h[k])

    def test_broadcasting3(self):
        np.random.seed(123)
        N = 16
        b = np.random.rand(3, N)
        for whole in [False, True]:
            for worN in [N, np.linspace(0, 1, N)]:
                w, h = freqz(b, worN=worN, whole=whole)
                assert_equal(w.size, N)
                for k in range(N):
                    bk = b[:, k]
                    ww, hh = freqz(bk, worN=w[k], whole=whole)
                    assert_allclose(ww, w[k])
                    assert_allclose(hh, h[k])

    def test_broadcasting4(self):
        np.random.seed(123)
        b = np.random.rand(4, 2, 1, 1)
        a = np.random.rand(5, 2, 1, 1)
        for whole in [False, True]:
            for worN in [np.random.rand(6, 7), np.empty((6, 0))]:
                w, h = freqz(b, a, worN=worN, whole=whole)
                assert_allclose(w, worN, rtol=1e-14)
                assert_equal(h.shape, (2,) + worN.shape)
                for k in range(2):
                    ww, hh = freqz(b[:, k, 0, 0], a[:, k, 0, 0], worN=worN.ravel(), whole=whole)
                    assert_allclose(ww, worN.ravel(), rtol=1e-14)
                    assert_allclose(hh, h[k, :, :].ravel())

    def test_backward_compat(self):
        w1, h1 = freqz([1.0], 1)
        w2, h2 = freqz([1.0], 1, None)
        assert_array_almost_equal(w1, w2)
        assert_array_almost_equal(h1, h2)

    def test_fs_param(self):
        fs = 900
        b = [0.03947915567748437, 0.11843746703245311, 0.11843746703245311, 0.03947915567748437]
        a = [1.0, -1.3199152021838287, 0.8034199108193842, -0.1676714632156805]
        w1, h1 = freqz(b, a, fs=fs)
        w2, h2 = freqz(b, a)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs / 2, 512, endpoint=False))
        w1, h1 = freqz(b, a, whole=True, fs=fs)
        w2, h2 = freqz(b, a, whole=True)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs, 512, endpoint=False))
        w1, h1 = freqz(b, a, 5, fs=fs)
        w2, h2 = freqz(b, a, 5)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs / 2, 5, endpoint=False))
        w1, h1 = freqz(b, a, 5, whole=True, fs=fs)
        w2, h2 = freqz(b, a, 5, whole=True)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs, 5, endpoint=False))
        for w in ([123], (123,), np.array([123]), (50, 123, 230), np.array([50, 123, 230])):
            w1, h1 = freqz(b, a, w, fs=fs)
            w2, h2 = freqz(b, a, 2 * pi * np.array(w) / fs)
            assert_allclose(h1, h2)
            assert_allclose(w, w1)

    def test_w_or_N_types(self):
        for N in (7, np.int8(7), np.int16(7), np.int32(7), np.int64(7), np.array(7), 8, np.int8(8), np.int16(8), np.int32(8), np.int64(8), np.array(8)):
            w, h = freqz([1.0], worN=N)
            assert_array_almost_equal(w, np.pi * np.arange(N) / N)
            assert_array_almost_equal(h, np.ones(N))
            w, h = freqz([1.0], worN=N, fs=100)
            assert_array_almost_equal(w, np.linspace(0, 50, N, endpoint=False))
            assert_array_almost_equal(h, np.ones(N))
        for w in (8.0, 8.0 + 0j):
            w_out, h = freqz([1.0], worN=w, fs=100)
            assert_array_almost_equal(w_out, [8])
            assert_array_almost_equal(h, [1])

    def test_nyquist(self):
        w, h = freqz([1.0], worN=8, include_nyquist=True)
        assert_array_almost_equal(w, np.pi * np.arange(8) / 7.0)
        assert_array_almost_equal(h, np.ones(8))
        w, h = freqz([1.0], worN=9, include_nyquist=True)
        assert_array_almost_equal(w, np.pi * np.arange(9) / 8.0)
        assert_array_almost_equal(h, np.ones(9))
        for a in [1, np.ones(2)]:
            w, h = freqz(np.ones(2), a, worN=0, include_nyquist=True)
            assert_equal(w.shape, (0,))
            assert_equal(h.shape, (0,))
            assert_equal(h.dtype, np.dtype('complex128'))
        w1, h1 = freqz([1.0], worN=8, whole=True, include_nyquist=True)
        w2, h2 = freqz([1.0], worN=8, whole=True, include_nyquist=False)
        assert_array_almost_equal(w1, w2)
        assert_array_almost_equal(h1, h2)

    @pytest.mark.parametrize('whole,nyquist,worN', [(False, False, 32), (False, True, 32), (True, False, 32), (True, True, 32), (False, False, 257), (False, True, 257), (True, False, 257), (True, True, 257)])
    def test_17289(self, whole, nyquist, worN):
        d = [0, 1]
        w, Drfft = freqz(d, worN=32, whole=whole, include_nyquist=nyquist)
        _, Dpoly = freqz(d, worN=w)
        assert_allclose(Drfft, Dpoly)