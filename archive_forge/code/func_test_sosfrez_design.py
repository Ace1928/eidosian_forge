import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_sosfrez_design(self):
    N, Wn = cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 60)
    sos = cheby2(N, 60, Wn, 'stop', output='sos')
    w, h = sosfreqz(sos)
    h = np.abs(h)
    w /= np.pi
    assert_allclose(20 * np.log10(h[w <= 0.1]), 0, atol=3.01)
    assert_allclose(20 * np.log10(h[w >= 0.6]), 0.0, atol=3.01)
    assert_allclose(h[(w >= 0.2) & (w <= 0.5)], 0.0, atol=0.001)
    N, Wn = cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 150)
    sos = cheby2(N, 150, Wn, 'stop', output='sos')
    w, h = sosfreqz(sos)
    dB = 20 * np.log10(np.abs(h))
    w /= np.pi
    assert_allclose(dB[w <= 0.1], 0, atol=3.01)
    assert_allclose(dB[w >= 0.6], 0.0, atol=3.01)
    assert_array_less(dB[(w >= 0.2) & (w <= 0.5)], -149.9)
    N, Wn = cheb1ord(0.2, 0.3, 3, 40)
    sos = cheby1(N, 3, Wn, 'low', output='sos')
    w, h = sosfreqz(sos)
    h = np.abs(h)
    w /= np.pi
    assert_allclose(20 * np.log10(h[w <= 0.2]), 0, atol=3.01)
    assert_allclose(h[w >= 0.3], 0.0, atol=0.01)
    N, Wn = cheb1ord(0.2, 0.3, 1, 150)
    sos = cheby1(N, 1, Wn, 'low', output='sos')
    w, h = sosfreqz(sos)
    dB = 20 * np.log10(np.abs(h))
    w /= np.pi
    assert_allclose(dB[w <= 0.2], 0, atol=1.01)
    assert_array_less(dB[w >= 0.3], -149.9)
    N, Wn = ellipord(0.3, 0.2, 3, 60)
    sos = ellip(N, 0.3, 60, Wn, 'high', output='sos')
    w, h = sosfreqz(sos)
    h = np.abs(h)
    w /= np.pi
    assert_allclose(20 * np.log10(h[w >= 0.3]), 0, atol=3.01)
    assert_allclose(h[w <= 0.1], 0.0, atol=0.0015)
    N, Wn = buttord([0.2, 0.5], [0.14, 0.6], 3, 40)
    sos = butter(N, Wn, 'band', output='sos')
    w, h = sosfreqz(sos)
    h = np.abs(h)
    w /= np.pi
    assert_allclose(h[w <= 0.14], 0.0, atol=0.01)
    assert_allclose(h[w >= 0.6], 0.0, atol=0.01)
    assert_allclose(20 * np.log10(h[(w >= 0.2) & (w <= 0.5)]), 0, atol=3.01)
    N, Wn = buttord([0.2, 0.5], [0.14, 0.6], 3, 100)
    sos = butter(N, Wn, 'band', output='sos')
    w, h = sosfreqz(sos)
    dB = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
    w /= np.pi
    assert_array_less(dB[(w > 0) & (w <= 0.14)], -99.9)
    assert_array_less(dB[w >= 0.6], -99.9)
    assert_allclose(dB[(w >= 0.2) & (w <= 0.5)], 0, atol=3.01)