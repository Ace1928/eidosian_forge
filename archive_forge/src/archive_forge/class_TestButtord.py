import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestButtord:

    def test_lowpass(self):
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 60
        N, Wn = buttord(wp, ws, rp, rs, False)
        b, a = butter(N, Wn, 'lowpass', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp, dB(h[w <= wp]))
        assert_array_less(dB(h[ws <= w]), -rs)
        assert_equal(N, 16)
        assert_allclose(Wn, 0.20002776782743284, rtol=1e-15)

    def test_highpass(self):
        wp = 0.3
        ws = 0.2
        rp = 3
        rs = 70
        N, Wn = buttord(wp, ws, rp, rs, False)
        b, a = butter(N, Wn, 'highpass', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp, dB(h[wp <= w]))
        assert_array_less(dB(h[w <= ws]), -rs)
        assert_equal(N, 18)
        assert_allclose(Wn, 0.2999660307913267, rtol=1e-15)

    def test_bandpass(self):
        wp = [0.2, 0.5]
        ws = [0.1, 0.6]
        rp = 3
        rs = 80
        N, Wn = buttord(wp, ws, rp, rs, False)
        b, a = butter(N, Wn, 'bandpass', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1, dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]), -rs + 0.1)
        assert_equal(N, 18)
        assert_allclose(Wn, [0.19998742411409134, 0.5000213959567628], rtol=1e-15)

    def test_bandstop(self):
        wp = [0.1, 0.6]
        ws = [0.2, 0.5]
        rp = 3
        rs = 90
        N, Wn = buttord(wp, ws, rp, rs, False)
        b, a = butter(N, Wn, 'bandstop', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp, dB(h[np.logical_or(w <= wp[0], wp[1] <= w)]))
        assert_array_less(dB(h[np.logical_and(ws[0] <= w, w <= ws[1])]), -rs)
        assert_equal(N, 20)
        assert_allclose(Wn, [0.14759432329294042, 0.5999736598527641], rtol=1e-06)

    def test_analog(self):
        wp = 200
        ws = 600
        rp = 3
        rs = 60
        N, Wn = buttord(wp, ws, rp, rs, True)
        b, a = butter(N, Wn, 'lowpass', True)
        w, h = freqs(b, a)
        assert_array_less(-rp, dB(h[w <= wp]))
        assert_array_less(dB(h[ws <= w]), -rs)
        assert_equal(N, 7)
        assert_allclose(Wn, 200.06785355671877, rtol=1e-15)
        n, Wn = buttord(1, 550 / 450, 1, 26, analog=True)
        assert_equal(n, 19)
        assert_allclose(Wn, 1.0361980524629517, rtol=1e-15)
        assert_equal(buttord(1, 1.2, 1, 80, analog=True)[0], 55)

    def test_fs_param(self):
        wp = [4410, 11025]
        ws = [2205, 13230]
        rp = 3
        rs = 80
        fs = 44100
        N, Wn = buttord(wp, ws, rp, rs, False, fs=fs)
        b, a = butter(N, Wn, 'bandpass', False, fs=fs)
        w, h = freqz(b, a, fs=fs)
        assert_array_less(-rp - 0.1, dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]), -rs + 0.1)
        assert_equal(N, 18)
        assert_allclose(Wn, [4409.722701715714, 11025.47178084662], rtol=1e-15)

    def test_invalid_input(self):
        with pytest.raises(ValueError) as exc_info:
            buttord([20, 50], [14, 60], 3, 2)
        assert 'gpass should be smaller than gstop' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            buttord([20, 50], [14, 60], -1, 2)
        assert 'gpass should be larger than 0.0' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            buttord([20, 50], [14, 60], 1, -2)
        assert 'gstop should be larger than 0.0' in str(exc_info.value)

    def test_runtime_warnings(self):
        msg = 'Order is zero.*|divide by zero encountered in divide'
        with pytest.warns(RuntimeWarning, match=msg):
            buttord(0.0, 1.0, 3, 60)

    def test_ellip_butter(self):
        n, wn = buttord([0.1, 0.6], [0.2, 0.5], 3, 60)
        assert n == 14