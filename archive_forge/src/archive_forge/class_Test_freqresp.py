from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
class Test_freqresp:

    def test_output_manual(self):
        system = lti([1], [1, 1])
        w = [0.1, 1, 10]
        w, H = freqresp(system, w=w)
        expected_re = [0.99, 0.5, 0.0099]
        expected_im = [-0.099, -0.5, -0.099]
        assert_almost_equal(H.real, expected_re, decimal=1)
        assert_almost_equal(H.imag, expected_im, decimal=1)

    def test_output(self):
        system = lti([1], [1, 1])
        w = [0.1, 1, 10, 100]
        w, H = freqresp(system, w=w)
        s = w * 1j
        expected = np.polyval(system.num, s) / np.polyval(system.den, s)
        assert_almost_equal(H.real, expected.real)
        assert_almost_equal(H.imag, expected.imag)

    def test_freq_range(self):
        system = lti([1], [1, 1])
        n = 10
        expected_w = np.logspace(-2, 1, n)
        w, H = freqresp(system, n=n)
        assert_almost_equal(w, expected_w)

    def test_pole_zero(self):
        system = lti([1], [1, 0])
        w, H = freqresp(system, n=2)
        assert_equal(w[0], 0.01)

    def test_from_state_space(self):
        a = np.array([1.0, 2.0, 2.0, 1.0])
        A = linalg.companion(a).T
        B = np.array([[0.0], [0.0], [1.0]])
        C = np.array([[1.0, 0.0, 0.0]])
        D = np.array([[0.0]])
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            system = lti(A, B, C, D)
            w, H = freqresp(system, n=100)
        s = w * 1j
        expected = 1.0 / (1.0 + 2 * s + 2 * s ** 2 + s ** 3)
        assert_almost_equal(H.real, expected.real)
        assert_almost_equal(H.imag, expected.imag)

    def test_from_zpk(self):
        system = lti([], [-1] * 4, [1])
        w = [0.1, 1, 10, 100]
        w, H = freqresp(system, w=w)
        s = w * 1j
        expected = 1 / (s + 1) ** 4
        assert_almost_equal(H.real, expected.real)
        assert_almost_equal(H.imag, expected.imag)