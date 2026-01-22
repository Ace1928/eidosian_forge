import warnings
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
import numpy as np
from numpy import array, float64
from multiprocessing.pool import ThreadPool
from scipy import optimize, linalg
from scipy.special import lambertw
from scipy.optimize._minpack_py import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning
from scipy.optimize._minimize import Bounds
class TestFSolve:

    def test_pressure_network_no_gradient(self):
        k = np.full(4, 0.5)
        Qtot = 4
        initial_guess = array([2.0, 0.0, 2.0, 0.0])
        final_flows, info, ier, mesg = optimize.fsolve(pressure_network, initial_guess, args=(Qtot, k), full_output=True)
        assert_array_almost_equal(final_flows, np.ones(4))
        assert_(ier == 1, mesg)

    def test_pressure_network_with_gradient(self):
        k = np.full(4, 0.5)
        Qtot = 4
        initial_guess = array([2.0, 0.0, 2.0, 0.0])
        final_flows = optimize.fsolve(pressure_network, initial_guess, args=(Qtot, k), fprime=pressure_network_jacobian)
        assert_array_almost_equal(final_flows, np.ones(4))

    def test_wrong_shape_func_callable(self):
        func = ReturnShape(1)
        x0 = [1.5, 2.0]
        assert_raises(TypeError, optimize.fsolve, func, x0)

    def test_wrong_shape_func_function(self):
        x0 = [1.5, 2.0]
        assert_raises(TypeError, optimize.fsolve, dummy_func, x0, args=((1,),))

    def test_wrong_shape_fprime_callable(self):
        func = ReturnShape(1)
        deriv_func = ReturnShape((2, 2))
        assert_raises(TypeError, optimize.fsolve, func, x0=[0, 1], fprime=deriv_func)

    def test_wrong_shape_fprime_function(self):

        def func(x):
            return dummy_func(x, (2,))

        def deriv_func(x):
            return dummy_func(x, (3, 3))
        assert_raises(TypeError, optimize.fsolve, func, x0=[0, 1], fprime=deriv_func)

    def test_func_can_raise(self):

        def func(*args):
            raise ValueError('I raised')
        with assert_raises(ValueError, match='I raised'):
            optimize.fsolve(func, x0=[0])

    def test_Dfun_can_raise(self):

        def func(x):
            return x - np.array([10])

        def deriv_func(*args):
            raise ValueError('I raised')
        with assert_raises(ValueError, match='I raised'):
            optimize.fsolve(func, x0=[0], fprime=deriv_func)

    def test_float32(self):

        def func(x):
            return np.array([x[0] - 100, x[1] - 1000], dtype=np.float32) ** 2
        p = optimize.fsolve(func, np.array([1, 1], np.float32))
        assert_allclose(func(p), [0, 0], atol=0.001)

    def test_reentrant_func(self):

        def func(*args):
            self.test_pressure_network_no_gradient()
            return pressure_network(*args)
        k = np.full(4, 0.5)
        Qtot = 4
        initial_guess = array([2.0, 0.0, 2.0, 0.0])
        final_flows, info, ier, mesg = optimize.fsolve(func, initial_guess, args=(Qtot, k), full_output=True)
        assert_array_almost_equal(final_flows, np.ones(4))
        assert_(ier == 1, mesg)

    def test_reentrant_Dfunc(self):

        def deriv_func(*args):
            self.test_pressure_network_with_gradient()
            return pressure_network_jacobian(*args)
        k = np.full(4, 0.5)
        Qtot = 4
        initial_guess = array([2.0, 0.0, 2.0, 0.0])
        final_flows = optimize.fsolve(pressure_network, initial_guess, args=(Qtot, k), fprime=deriv_func)
        assert_array_almost_equal(final_flows, np.ones(4))

    def test_concurrent_no_gradient(self):
        v = sequence_parallel([self.test_pressure_network_no_gradient] * 10)
        assert all([result is None for result in v])

    def test_concurrent_with_gradient(self):
        v = sequence_parallel([self.test_pressure_network_with_gradient] * 10)
        assert all([result is None for result in v])