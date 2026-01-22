import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
class TestChirp:

    def test_linear_at_zero(self):
        w = waveforms.chirp(t=0, f0=1.0, f1=2.0, t1=1.0, method='linear')
        assert_almost_equal(w, 1.0)

    def test_linear_freq_01(self):
        method = 'linear'
        f0 = 1.0
        f1 = 2.0
        t1 = 1.0
        t = np.linspace(0, t1, 100)
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        tf, f = compute_frequency(t, phase)
        abserr = np.max(np.abs(f - chirp_linear(tf, f0, f1, t1)))
        assert_(abserr < 1e-06)

    def test_linear_freq_02(self):
        method = 'linear'
        f0 = 200.0
        f1 = 100.0
        t1 = 10.0
        t = np.linspace(0, t1, 100)
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        tf, f = compute_frequency(t, phase)
        abserr = np.max(np.abs(f - chirp_linear(tf, f0, f1, t1)))
        assert_(abserr < 1e-06)

    def test_quadratic_at_zero(self):
        w = waveforms.chirp(t=0, f0=1.0, f1=2.0, t1=1.0, method='quadratic')
        assert_almost_equal(w, 1.0)

    def test_quadratic_at_zero2(self):
        w = waveforms.chirp(t=0, f0=1.0, f1=2.0, t1=1.0, method='quadratic', vertex_zero=False)
        assert_almost_equal(w, 1.0)

    def test_quadratic_freq_01(self):
        method = 'quadratic'
        f0 = 1.0
        f1 = 2.0
        t1 = 1.0
        t = np.linspace(0, t1, 2000)
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        tf, f = compute_frequency(t, phase)
        abserr = np.max(np.abs(f - chirp_quadratic(tf, f0, f1, t1)))
        assert_(abserr < 1e-06)

    def test_quadratic_freq_02(self):
        method = 'quadratic'
        f0 = 20.0
        f1 = 10.0
        t1 = 10.0
        t = np.linspace(0, t1, 2000)
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        tf, f = compute_frequency(t, phase)
        abserr = np.max(np.abs(f - chirp_quadratic(tf, f0, f1, t1)))
        assert_(abserr < 1e-06)

    def test_logarithmic_at_zero(self):
        w = waveforms.chirp(t=0, f0=1.0, f1=2.0, t1=1.0, method='logarithmic')
        assert_almost_equal(w, 1.0)

    def test_logarithmic_freq_01(self):
        method = 'logarithmic'
        f0 = 1.0
        f1 = 2.0
        t1 = 1.0
        t = np.linspace(0, t1, 10000)
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        tf, f = compute_frequency(t, phase)
        abserr = np.max(np.abs(f - chirp_geometric(tf, f0, f1, t1)))
        assert_(abserr < 1e-06)

    def test_logarithmic_freq_02(self):
        method = 'logarithmic'
        f0 = 200.0
        f1 = 100.0
        t1 = 10.0
        t = np.linspace(0, t1, 10000)
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        tf, f = compute_frequency(t, phase)
        abserr = np.max(np.abs(f - chirp_geometric(tf, f0, f1, t1)))
        assert_(abserr < 1e-06)

    def test_logarithmic_freq_03(self):
        method = 'logarithmic'
        f0 = 100.0
        f1 = 100.0
        t1 = 10.0
        t = np.linspace(0, t1, 10000)
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        tf, f = compute_frequency(t, phase)
        abserr = np.max(np.abs(f - chirp_geometric(tf, f0, f1, t1)))
        assert_(abserr < 1e-06)

    def test_hyperbolic_at_zero(self):
        w = waveforms.chirp(t=0, f0=10.0, f1=1.0, t1=1.0, method='hyperbolic')
        assert_almost_equal(w, 1.0)

    def test_hyperbolic_freq_01(self):
        method = 'hyperbolic'
        t1 = 1.0
        t = np.linspace(0, t1, 10000)
        cases = [[10.0, 1.0], [1.0, 10.0], [-10.0, -1.0], [-1.0, -10.0]]
        for f0, f1 in cases:
            phase = waveforms._chirp_phase(t, f0, t1, f1, method)
            tf, f = compute_frequency(t, phase)
            expected = chirp_hyperbolic(tf, f0, f1, t1)
            assert_allclose(f, expected)

    def test_hyperbolic_zero_freq(self):
        method = 'hyperbolic'
        t1 = 1.0
        t = np.linspace(0, t1, 5)
        assert_raises(ValueError, waveforms.chirp, t, 0, t1, 1, method)
        assert_raises(ValueError, waveforms.chirp, t, 1, t1, 0, method)

    def test_unknown_method(self):
        method = 'foo'
        f0 = 10.0
        f1 = 20.0
        t1 = 1.0
        t = np.linspace(0, t1, 10)
        assert_raises(ValueError, waveforms.chirp, t, f0, t1, f1, method)

    def test_integer_t1(self):
        f0 = 10.0
        f1 = 20.0
        t = np.linspace(-1, 1, 11)
        t1 = 3.0
        float_result = waveforms.chirp(t, f0, t1, f1)
        t1 = 3
        int_result = waveforms.chirp(t, f0, t1, f1)
        err_msg = "Integer input 't1=3' gives wrong result"
        assert_equal(int_result, float_result, err_msg=err_msg)

    def test_integer_f0(self):
        f1 = 20.0
        t1 = 3.0
        t = np.linspace(-1, 1, 11)
        f0 = 10.0
        float_result = waveforms.chirp(t, f0, t1, f1)
        f0 = 10
        int_result = waveforms.chirp(t, f0, t1, f1)
        err_msg = "Integer input 'f0=10' gives wrong result"
        assert_equal(int_result, float_result, err_msg=err_msg)

    def test_integer_f1(self):
        f0 = 10.0
        t1 = 3.0
        t = np.linspace(-1, 1, 11)
        f1 = 20.0
        float_result = waveforms.chirp(t, f0, t1, f1)
        f1 = 20
        int_result = waveforms.chirp(t, f0, t1, f1)
        err_msg = "Integer input 'f1=20' gives wrong result"
        assert_equal(int_result, float_result, err_msg=err_msg)

    def test_integer_all(self):
        f0 = 10
        t1 = 3
        f1 = 20
        t = np.linspace(-1, 1, 11)
        float_result = waveforms.chirp(t, float(f0), float(t1), float(f1))
        int_result = waveforms.chirp(t, f0, t1, f1)
        err_msg = "Integer input 'f0=10, t1=3, f1=20' gives wrong result"
        assert_equal(int_result, float_result, err_msg=err_msg)