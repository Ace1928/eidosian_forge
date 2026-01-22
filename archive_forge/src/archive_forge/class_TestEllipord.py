import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestEllipord:

    def test_lowpass(self):
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 60
        N, Wn = ellipord(wp, ws, rp, rs, False)
        b, a = ellip(N, rp, rs, Wn, 'lp', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1, dB(h[w <= wp]))
        assert_array_less(dB(h[ws <= w]), -rs + 0.1)
        assert_equal(N, 5)
        assert_allclose(Wn, 0.2, rtol=1e-15)

    def test_lowpass_1000dB(self):
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 1000
        N, Wn = ellipord(wp, ws, rp, rs, False)
        sos = ellip(N, rp, rs, Wn, 'lp', False, output='sos')
        w, h = sosfreqz(sos)
        w /= np.pi
        assert_array_less(-rp - 0.1, dB(h[w <= wp]))
        assert_array_less(dB(h[ws <= w]), -rs + 0.1)

    def test_highpass(self):
        wp = 0.3
        ws = 0.2
        rp = 3
        rs = 70
        N, Wn = ellipord(wp, ws, rp, rs, False)
        b, a = ellip(N, rp, rs, Wn, 'hp', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1, dB(h[wp <= w]))
        assert_array_less(dB(h[w <= ws]), -rs + 0.1)
        assert_equal(N, 6)
        assert_allclose(Wn, 0.3, rtol=1e-15)

    def test_bandpass(self):
        wp = [0.2, 0.5]
        ws = [0.1, 0.6]
        rp = 3
        rs = 80
        N, Wn = ellipord(wp, ws, rp, rs, False)
        b, a = ellip(N, rp, rs, Wn, 'bp', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1, dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]), -rs + 0.1)
        assert_equal(N, 6)
        assert_allclose(Wn, [0.2, 0.5], rtol=1e-15)

    def test_bandstop(self):
        wp = [0.1, 0.6]
        ws = [0.2, 0.5]
        rp = 3
        rs = 90
        N, Wn = ellipord(wp, ws, rp, rs, False)
        b, a = ellip(N, rp, rs, Wn, 'bs', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1, dB(h[np.logical_or(w <= wp[0], wp[1] <= w)]))
        assert_array_less(dB(h[np.logical_and(ws[0] <= w, w <= ws[1])]), -rs + 0.1)
        assert_equal(N, 7)
        assert_allclose(Wn, [0.14758232794342988, 0.6], rtol=1e-05)

    def test_analog(self):
        wp = [1000, 6000]
        ws = [2000, 5000]
        rp = 3
        rs = 90
        N, Wn = ellipord(wp, ws, rp, rs, True)
        b, a = ellip(N, rp, rs, Wn, 'bs', True)
        w, h = freqs(b, a)
        assert_array_less(-rp - 0.1, dB(h[np.logical_or(w <= wp[0], wp[1] <= w)]))
        assert_array_less(dB(h[np.logical_and(ws[0] <= w, w <= ws[1])]), -rs + 0.1)
        assert_equal(N, 8)
        assert_allclose(Wn, [1666.6666, 6000])
        assert_equal(ellipord(1, 1.2, 1, 80, analog=True)[0], 9)

    def test_fs_param(self):
        wp = [400, 2400]
        ws = [800, 2000]
        rp = 3
        rs = 90
        fs = 8000
        N, Wn = ellipord(wp, ws, rp, rs, False, fs=fs)
        b, a = ellip(N, rp, rs, Wn, 'bs', False, fs=fs)
        w, h = freqz(b, a, fs=fs)
        assert_array_less(-rp - 0.1, dB(h[np.logical_or(w <= wp[0], wp[1] <= w)]))
        assert_array_less(dB(h[np.logical_and(ws[0] <= w, w <= ws[1])]), -rs + 0.1)
        assert_equal(N, 7)
        assert_allclose(Wn, [590.3293117737195, 2400], rtol=1e-05)

    def test_invalid_input(self):
        with pytest.raises(ValueError) as exc_info:
            ellipord(0.2, 0.5, 3, 2)
        assert 'gpass should be smaller than gstop' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            ellipord(0.2, 0.5, -1, 2)
        assert 'gpass should be larger than 0.0' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            ellipord(0.2, 0.5, 1, -2)
        assert 'gstop should be larger than 0.0' in str(exc_info.value)

    def test_ellip_butter(self):
        n, wn = ellipord([0.1, 0.6], [0.2, 0.5], 3, 60)
        assert n == 5