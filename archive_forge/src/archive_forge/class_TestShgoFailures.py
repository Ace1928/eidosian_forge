import logging
import sys
import numpy
import numpy as np
import time
from multiprocessing import Pool
from numpy.testing import assert_allclose, IS_PYPY
import pytest
from pytest import raises as assert_raises, warns
from scipy.optimize import (shgo, Bounds, minimize_scalar, minimize, rosen,
from scipy.optimize._constraints import new_constraint_to_old
from scipy.optimize._shgo import SHGO
class TestShgoFailures:

    def test_1_maxiter(self):
        """Test failure on insufficient iterations"""
        options = {'maxiter': 2}
        res = shgo(test4_1.f, test4_1.bounds, n=2, iters=None, options=options, sampling_method='sobol')
        numpy.testing.assert_equal(False, res.success)
        numpy.testing.assert_equal(4, res.tnev)

    def test_2_sampling(self):
        """Rejection of unknown sampling method"""
        assert_raises(ValueError, shgo, test1_1.f, test1_1.bounds, sampling_method='not_Sobol')

    def test_3_1_no_min_pool_sobol(self):
        """Check that the routine stops when no minimiser is found
           after maximum specified function evaluations"""
        options = {'maxfev': 10, 'disp': True}
        res = shgo(test_table.f, test_table.bounds, n=3, options=options, sampling_method='sobol')
        numpy.testing.assert_equal(False, res.success)
        numpy.testing.assert_equal(12, res.nfev)

    def test_3_2_no_min_pool_simplicial(self):
        """Check that the routine stops when no minimiser is found
           after maximum specified sampling evaluations"""
        options = {'maxev': 10, 'disp': True}
        res = shgo(test_table.f, test_table.bounds, n=3, options=options, sampling_method='simplicial')
        numpy.testing.assert_equal(False, res.success)

    def test_4_1_bound_err(self):
        """Specified bounds ub > lb"""
        bounds = [(6, 3), (3, 5)]
        assert_raises(ValueError, shgo, test1_1.f, bounds)

    def test_4_2_bound_err(self):
        """Specified bounds are of the form (lb, ub)"""
        bounds = [(3, 5, 5), (3, 5)]
        assert_raises(ValueError, shgo, test1_1.f, bounds)

    def test_5_1_1_infeasible_sobol(self):
        """Ensures the algorithm terminates on infeasible problems
           after maxev is exceeded. Use infty constraints option"""
        options = {'maxev': 100, 'disp': True}
        res = shgo(test_infeasible.f, test_infeasible.bounds, constraints=test_infeasible.cons, n=100, options=options, sampling_method='sobol')
        numpy.testing.assert_equal(False, res.success)

    def test_5_1_2_infeasible_sobol(self):
        """Ensures the algorithm terminates on infeasible problems
           after maxev is exceeded. Do not use infty constraints option"""
        options = {'maxev': 100, 'disp': True, 'infty_constraints': False}
        res = shgo(test_infeasible.f, test_infeasible.bounds, constraints=test_infeasible.cons, n=100, options=options, sampling_method='sobol')
        numpy.testing.assert_equal(False, res.success)

    def test_5_2_infeasible_simplicial(self):
        """Ensures the algorithm terminates on infeasible problems
           after maxev is exceeded."""
        options = {'maxev': 1000, 'disp': False}
        res = shgo(test_infeasible.f, test_infeasible.bounds, constraints=test_infeasible.cons, n=100, options=options, sampling_method='simplicial')
        numpy.testing.assert_equal(False, res.success)

    def test_6_1_lower_known_f_min(self):
        """Test Global mode limiting local evaluations with f* too high"""
        options = {'f_min': test2_1.expected_fun + 2.0, 'f_tol': 1e-06, 'minimize_every_iter': True, 'local_iter': 1, 'infty_constraints': False}
        args = (test2_1.f, test2_1.bounds)
        kwargs = {'constraints': test2_1.cons, 'n': None, 'iters': None, 'options': options, 'sampling_method': 'sobol'}
        warns(UserWarning, shgo, *args, **kwargs)

    def test(self):
        from scipy.optimize import rosen, shgo
        bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]

        def fun(x):
            fun.nfev += 1
            return rosen(x)
        fun.nfev = 0
        result = shgo(fun, bounds)
        print(result.x, result.fun, fun.nfev)