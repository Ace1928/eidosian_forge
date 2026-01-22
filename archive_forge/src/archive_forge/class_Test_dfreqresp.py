import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
class Test_dfreqresp:

    def test_manual(self):
        system = TransferFunction(1, [1, -0.2], dt=0.1)
        w = [0.1, 1, 10]
        w, H = dfreqresp(system, w=w)
        expected_re = [1.2383, 0.413, -0.7553]
        assert_almost_equal(H.real, expected_re, decimal=4)
        expected_im = [-0.1555, -1.0214, 0.3955]
        assert_almost_equal(H.imag, expected_im, decimal=4)

    def test_auto(self):
        system = TransferFunction(1, [1, -0.2], dt=0.1)
        w = [0.1, 1, 10, 100]
        w, H = dfreqresp(system, w=w)
        jw = np.exp(w * 1j)
        y = np.polyval(system.num, jw) / np.polyval(system.den, jw)
        expected_re = y.real
        assert_almost_equal(H.real, expected_re)
        expected_im = y.imag
        assert_almost_equal(H.imag, expected_im)

    def test_freq_range(self):
        system = TransferFunction(1, [1, -0.2], dt=0.1)
        n = 10
        expected_w = np.linspace(0, np.pi, 10, endpoint=False)
        w, H = dfreqresp(system, n=n)
        assert_almost_equal(w, expected_w)

    def test_pole_one(self):
        system = TransferFunction([1], [1, -1], dt=0.1)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, message='divide by zero')
            sup.filter(RuntimeWarning, message='invalid value encountered')
            w, H = dfreqresp(system, n=2)
        assert_equal(w[0], 0.0)

    def test_error(self):
        system = lti([1], [1, 1])
        assert_raises(AttributeError, dfreqresp, system)

    def test_from_state_space(self):
        system_TF = dlti([2], [1, -0.5, 0, 0])
        A = np.array([[0.5, 0, 0], [1, 0, 0], [0, 1, 0]])
        B = np.array([[1, 0, 0]]).T
        C = np.array([[0, 0, 2]])
        D = 0
        system_SS = dlti(A, B, C, D)
        w = 10.0 ** np.arange(-3, 0, 0.5)
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            w1, H1 = dfreqresp(system_TF, w=w)
            w2, H2 = dfreqresp(system_SS, w=w)
        assert_almost_equal(H1, H2)

    def test_from_zpk(self):
        system_ZPK = dlti([], [0.2], 0.3)
        system_TF = dlti(0.3, [1, -0.2])
        w = [0.1, 1, 10, 100]
        w1, H1 = dfreqresp(system_ZPK, w=w)
        w2, H2 = dfreqresp(system_TF, w=w)
        assert_almost_equal(H1, H2)