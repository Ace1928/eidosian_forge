from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
class _TestLsimFuncs:
    digits_accuracy = 7

    @abstractmethod
    def func(self, *args, **kwargs):
        pass

    def lti_nowarn(self, *args):
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            system = lti(*args)
        return system

    def test_first_order(self):
        system = self.lti_nowarn(-1.0, 1.0, 1.0, 0.0)
        t = np.linspace(0, 5)
        u = np.zeros_like(t)
        tout, y, x = self.func(system, u, t, X0=[1.0])
        expected_x = np.exp(-tout)
        assert_almost_equal(x, expected_x)
        assert_almost_equal(y, expected_x)

    def test_second_order(self):
        t = np.linspace(0, 10, 1001)
        u = np.zeros_like(t)
        system = self.lti_nowarn([1.0], [1.0, 2.0, 1.0])
        tout, y, x = self.func(system, u, t, X0=[1.0, 0.0])
        expected_x = (1.0 - tout) * np.exp(-tout)
        assert_almost_equal(x[:, 0], expected_x)

    def test_integrator(self):
        system = self.lti_nowarn(0.0, 1.0, 1.0, 0.0)
        t = np.linspace(0, 5)
        u = t
        tout, y, x = self.func(system, u, t)
        expected_x = 0.5 * tout ** 2
        assert_almost_equal(x, expected_x, decimal=self.digits_accuracy)
        assert_almost_equal(y, expected_x, decimal=self.digits_accuracy)

    def test_two_states(self):
        A = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        C = np.array([1.0, 0.0])
        D = np.zeros((1, 2))
        system = self.lti_nowarn(A, B, C, D)
        t = np.linspace(0, 10.0, 21)
        u = np.zeros((len(t), 2))
        tout, y, x = self.func(system, U=u, T=t, X0=[1.0, 1.0])
        expected_y = np.exp(-tout)
        expected_x0 = np.exp(-tout)
        expected_x1 = np.exp(-2.0 * tout)
        assert_almost_equal(y, expected_y)
        assert_almost_equal(x[:, 0], expected_x0)
        assert_almost_equal(x[:, 1], expected_x1)

    def test_double_integrator(self):
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        C = np.array([[2.0, 0.0]])
        system = self.lti_nowarn(A, B, C, 0.0)
        t = np.linspace(0, 5)
        u = np.ones_like(t)
        tout, y, x = self.func(system, u, t)
        expected_x = np.transpose(np.array([0.5 * tout ** 2, tout]))
        expected_y = tout ** 2
        assert_almost_equal(x, expected_x, decimal=self.digits_accuracy)
        assert_almost_equal(y, expected_y, decimal=self.digits_accuracy)

    def test_jordan_block(self):
        A = np.array([[-1.0, 1.0], [0.0, -1.0]])
        B = np.array([[0.0], [1.0]])
        C = np.array([[1.0, 0.0]])
        system = self.lti_nowarn(A, B, C, 0.0)
        t = np.linspace(0, 5)
        u = np.zeros_like(t)
        tout, y, x = self.func(system, u, t, X0=[0.0, 1.0])
        expected_y = tout * np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_miso(self):
        A = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        C = np.array([1.0, 0.0])
        D = np.zeros((1, 2))
        system = self.lti_nowarn(A, B, C, D)
        t = np.linspace(0, 5.0, 101)
        u = np.zeros((len(t), 2))
        tout, y, x = self.func(system, u, t, X0=[1.0, 1.0])
        expected_y = np.exp(-tout)
        expected_x0 = np.exp(-tout)
        expected_x1 = np.exp(-2.0 * tout)
        assert_almost_equal(y, expected_y)
        assert_almost_equal(x[:, 0], expected_x0)
        assert_almost_equal(x[:, 1], expected_x1)