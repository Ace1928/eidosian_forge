import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
class TestLinprogRSCommon(LinprogRSTests):
    options = {}

    def test_cyclic_bland(self):
        pytest.skip('Intermittent failure acceptable.')

    def test_nontrivial_problem_with_guess(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_unbounded_variables(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        bounds = [(None, None), (None, None), (0, None), (None, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_bounded_variables(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        bounds = [(None, 1), (1, None), (0, None), (0.4, 0.6)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_negative_unbounded_variable(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        b_eq = [4]
        x_star = np.array([-219 / 385, 582 / 385, 0, 4 / 10])
        f_star = 3951 / 385
        bounds = [(None, None), (1, None), (0, None), (0.4, 0.6)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_bad_guess(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        bad_guess = [1, 2, 3, 0.5]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options, x0=bad_guess)
        assert_equal(res.status, 6)

    def test_redundant_constraints_with_guess(self):
        A, b, c, _, _ = magic_square(3)
        p = np.random.rand(*c.shape)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, 'A_eq does not appear...')
            sup.filter(RuntimeWarning, 'invalid value encountered')
            sup.filter(LinAlgWarning)
            res = linprog(c, A_eq=A, b_eq=b, method=self.method)
            res2 = linprog(c, A_eq=A, b_eq=b, method=self.method, x0=res.x)
            res3 = linprog(c + p, A_eq=A, b_eq=b, method=self.method, x0=res.x)
        _assert_success(res2, desired_fun=1.730550597)
        assert_equal(res2.nit, 0)
        _assert_success(res3)
        assert_(res3.nit < res.nit)