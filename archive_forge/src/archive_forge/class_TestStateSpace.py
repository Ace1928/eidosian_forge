from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
class TestStateSpace:

    def test_initialization(self):
        StateSpace(1, 1, 1, 1)
        StateSpace([1], [2], [3], [4])
        StateSpace(np.array([[1, 2], [3, 4]]), np.array([[1], [2]]), np.array([[1, 0]]), np.array([[0]]))

    def test_conversion(self):
        s = StateSpace(1, 2, 3, 4)
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))
        assert_(StateSpace(s) is not s)
        assert_(s.to_ss() is not s)

    def test_properties(self):
        s = StateSpace(1, 1, 1, 1)
        assert_equal(s.poles, [1])
        assert_equal(s.zeros, [0])
        assert_(s.dt is None)

    def test_operators(self):

        class BadType:
            pass
        s1 = StateSpace(np.array([[-0.5, 0.7], [0.3, -0.8]]), np.array([[1], [0]]), np.array([[1, 0]]), np.array([[0]]))
        s2 = StateSpace(np.array([[-0.2, -0.1], [0.4, -0.1]]), np.array([[1], [0]]), np.array([[1, 0]]), np.array([[0]]))
        s_discrete = s1.to_discrete(0.1)
        s2_discrete = s2.to_discrete(0.2)
        s3_discrete = s2.to_discrete(0.1)
        t = np.linspace(0, 1, 100)
        u = np.zeros_like(t)
        u[0] = 1
        for typ in (int, float, complex, np.float32, np.complex128, np.array):
            assert_allclose(lsim(typ(2) * s1, U=u, T=t)[1], typ(2) * lsim(s1, U=u, T=t)[1])
            assert_allclose(lsim(s1 * typ(2), U=u, T=t)[1], lsim(s1, U=u, T=t)[1] * typ(2))
            assert_allclose(lsim(s1 / typ(2), U=u, T=t)[1], lsim(s1, U=u, T=t)[1] / typ(2))
            with assert_raises(TypeError):
                typ(2) / s1
        assert_allclose(lsim(s1 * 2, U=u, T=t)[1], lsim(s1, U=2 * u, T=t)[1])
        assert_allclose(lsim(s1 * s2, U=u, T=t)[1], lsim(s1, U=lsim(s2, U=u, T=t)[1], T=t)[1], atol=1e-05)
        with assert_raises(TypeError):
            s1 / s1
        with assert_raises(TypeError):
            s1 * s_discrete
        with assert_raises(TypeError):
            s_discrete * s2_discrete
        with assert_raises(TypeError):
            s1 * BadType()
        with assert_raises(TypeError):
            BadType() * s1
        with assert_raises(TypeError):
            s1 / BadType()
        with assert_raises(TypeError):
            BadType() / s1
        assert_allclose(lsim(s1 + 2, U=u, T=t)[1], 2 * u + lsim(s1, U=u, T=t)[1])
        with assert_raises(ValueError):
            s1 + np.array([1, 2])
        with assert_raises(ValueError):
            np.array([1, 2]) + s1
        with assert_raises(TypeError):
            s1 + s_discrete
        with assert_raises(ValueError):
            s1 / np.array([[1, 2], [3, 4]])
        with assert_raises(TypeError):
            s_discrete + s2_discrete
        with assert_raises(TypeError):
            s1 + BadType()
        with assert_raises(TypeError):
            BadType() + s1
        assert_allclose(lsim(s1 + s2, U=u, T=t)[1], lsim(s1, U=u, T=t)[1] + lsim(s2, U=u, T=t)[1])
        assert_allclose(lsim(s1 - 2, U=u, T=t)[1], -2 * u + lsim(s1, U=u, T=t)[1])
        assert_allclose(lsim(2 - s1, U=u, T=t)[1], 2 * u + lsim(-s1, U=u, T=t)[1])
        assert_allclose(lsim(s1 - s2, U=u, T=t)[1], lsim(s1, U=u, T=t)[1] - lsim(s2, U=u, T=t)[1])
        with assert_raises(TypeError):
            s1 - BadType()
        with assert_raises(TypeError):
            BadType() - s1
        s = s_discrete + s3_discrete
        assert_(s.dt == 0.1)
        s = s_discrete * s3_discrete
        assert_(s.dt == 0.1)
        s = 3 * s_discrete
        assert_(s.dt == 0.1)
        s = -s_discrete
        assert_(s.dt == 0.1)