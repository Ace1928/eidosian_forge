import itertools
import platform
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy import optimize
from scipy.optimize._minimize import Bounds, NonlinearConstraint
from scipy.optimize._minimize import (MINIMIZE_METHODS,
from scipy.optimize._linprog import LINPROG_METHODS
from scipy.optimize._root import ROOT_METHODS
from scipy.optimize._root_scalar import ROOT_SCALAR_METHODS
from scipy.optimize._qap import QUADRATIC_ASSIGNMENT_METHODS
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
from scipy.optimize._optimize import MemoizeJac, show_options, OptimizeResult
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.sparse import (coo_matrix, csc_matrix, csr_matrix, coo_array,
class TestOptimizeScalar:

    def setup_method(self):
        self.solution = 1.5

    def fun(self, x, a=1.5):
        """Objective function"""
        return (x - a) ** 2 - 0.8

    def test_brent(self):
        x = optimize.brent(self.fun)
        assert_allclose(x, self.solution, atol=1e-06)
        x = optimize.brent(self.fun, brack=(-3, -2))
        assert_allclose(x, self.solution, atol=1e-06)
        x = optimize.brent(self.fun, full_output=True)
        assert_allclose(x[0], self.solution, atol=1e-06)
        x = optimize.brent(self.fun, brack=(-15, -1, 15))
        assert_allclose(x, self.solution, atol=1e-06)
        message = '\\(f\\(xb\\) < f\\(xa\\)\\) and \\(f\\(xb\\) < f\\(xc\\)\\)'
        with pytest.raises(ValueError, match=message):
            optimize.brent(self.fun, brack=(-1, 0, 1))
        message = '\\(xa < xb\\) and \\(xb < xc\\)'
        with pytest.raises(ValueError, match=message):
            optimize.brent(self.fun, brack=(0, -1, 1))

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_golden(self):
        x = optimize.golden(self.fun)
        assert_allclose(x, self.solution, atol=1e-06)
        x = optimize.golden(self.fun, brack=(-3, -2))
        assert_allclose(x, self.solution, atol=1e-06)
        x = optimize.golden(self.fun, full_output=True)
        assert_allclose(x[0], self.solution, atol=1e-06)
        x = optimize.golden(self.fun, brack=(-15, -1, 15))
        assert_allclose(x, self.solution, atol=1e-06)
        x = optimize.golden(self.fun, tol=0)
        assert_allclose(x, self.solution)
        maxiter_test_cases = [0, 1, 5]
        for maxiter in maxiter_test_cases:
            x0 = optimize.golden(self.fun, maxiter=0, full_output=True)
            x = optimize.golden(self.fun, maxiter=maxiter, full_output=True)
            nfev0, nfev = (x0[2], x[2])
            assert_equal(nfev - nfev0, maxiter)
        message = '\\(f\\(xb\\) < f\\(xa\\)\\) and \\(f\\(xb\\) < f\\(xc\\)\\)'
        with pytest.raises(ValueError, match=message):
            optimize.golden(self.fun, brack=(-1, 0, 1))
        message = '\\(xa < xb\\) and \\(xb < xc\\)'
        with pytest.raises(ValueError, match=message):
            optimize.golden(self.fun, brack=(0, -1, 1))

    def test_fminbound(self):
        x = optimize.fminbound(self.fun, 0, 1)
        assert_allclose(x, 1, atol=0.0001)
        x = optimize.fminbound(self.fun, 1, 5)
        assert_allclose(x, self.solution, atol=1e-06)
        x = optimize.fminbound(self.fun, np.array([1]), np.array([5]))
        assert_allclose(x, self.solution, atol=1e-06)
        assert_raises(ValueError, optimize.fminbound, self.fun, 5, 1)

    def test_fminbound_scalar(self):
        with pytest.raises(ValueError, match='.*must be finite scalars.*'):
            optimize.fminbound(self.fun, np.zeros((1, 2)), 1)
        x = optimize.fminbound(self.fun, 1, np.array(5))
        assert_allclose(x, self.solution, atol=1e-06)

    def test_gh11207(self):

        def fun(x):
            return x ** 2
        optimize.fminbound(fun, 0, 0)

    def test_minimize_scalar(self):
        x = optimize.minimize_scalar(self.fun).x
        assert_allclose(x, self.solution, atol=1e-06)
        x = optimize.minimize_scalar(self.fun, method='Brent')
        assert x.success
        x = optimize.minimize_scalar(self.fun, method='Brent', options=dict(maxiter=3))
        assert not x.success
        x = optimize.minimize_scalar(self.fun, bracket=(-3, -2), args=(1.5,), method='Brent').x
        assert_allclose(x, self.solution, atol=1e-06)
        x = optimize.minimize_scalar(self.fun, method='Brent', args=(1.5,)).x
        assert_allclose(x, self.solution, atol=1e-06)
        x = optimize.minimize_scalar(self.fun, bracket=(-15, -1, 15), args=(1.5,), method='Brent').x
        assert_allclose(x, self.solution, atol=1e-06)
        x = optimize.minimize_scalar(self.fun, bracket=(-3, -2), args=(1.5,), method='golden').x
        assert_allclose(x, self.solution, atol=1e-06)
        x = optimize.minimize_scalar(self.fun, method='golden', args=(1.5,)).x
        assert_allclose(x, self.solution, atol=1e-06)
        x = optimize.minimize_scalar(self.fun, bracket=(-15, -1, 15), args=(1.5,), method='golden').x
        assert_allclose(x, self.solution, atol=1e-06)
        x = optimize.minimize_scalar(self.fun, bounds=(0, 1), args=(1.5,), method='Bounded').x
        assert_allclose(x, 1, atol=0.0001)
        x = optimize.minimize_scalar(self.fun, bounds=(1, 5), args=(1.5,), method='bounded').x
        assert_allclose(x, self.solution, atol=1e-06)
        x = optimize.minimize_scalar(self.fun, bounds=(np.array([1]), np.array([5])), args=(np.array([1.5]),), method='bounded').x
        assert_allclose(x, self.solution, atol=1e-06)
        assert_raises(ValueError, optimize.minimize_scalar, self.fun, bounds=(5, 1), method='bounded', args=(1.5,))
        assert_raises(ValueError, optimize.minimize_scalar, self.fun, bounds=(np.zeros(2), 1), method='bounded', args=(1.5,))
        x = optimize.minimize_scalar(self.fun, bounds=(1, np.array(5)), method='bounded').x
        assert_allclose(x, self.solution, atol=1e-06)

    def test_minimize_scalar_custom(self):

        def custmin(fun, bracket, args=(), maxfev=None, stepsize=0.1, maxiter=100, callback=None, **options):
            bestx = (bracket[1] + bracket[0]) / 2.0
            besty = fun(bestx)
            funcalls = 1
            niter = 0
            improved = True
            stop = False
            while improved and (not stop) and (niter < maxiter):
                improved = False
                niter += 1
                for testx in [bestx - stepsize, bestx + stepsize]:
                    testy = fun(testx, *args)
                    funcalls += 1
                    if testy < besty:
                        besty = testy
                        bestx = testx
                        improved = True
                if callback is not None:
                    callback(bestx)
                if maxfev is not None and funcalls >= maxfev:
                    stop = True
                    break
            return optimize.OptimizeResult(fun=besty, x=bestx, nit=niter, nfev=funcalls, success=niter > 1)
        res = optimize.minimize_scalar(self.fun, bracket=(0, 4), method=custmin, options=dict(stepsize=0.05))
        assert_allclose(res.x, self.solution, atol=1e-06)

    def test_minimize_scalar_coerce_args_param(self):
        optimize.minimize_scalar(self.fun, args=1.5)

    @pytest.mark.parametrize('method', ['brent', 'bounded', 'golden'])
    def test_disp(self, method):
        for disp in [0, 1, 2, 3]:
            optimize.minimize_scalar(self.fun, options={'disp': disp})

    @pytest.mark.parametrize('method', ['brent', 'bounded', 'golden'])
    def test_result_attributes(self, method):
        kwargs = {'bounds': [-10, 10]} if method == 'bounded' else {}
        result = optimize.minimize_scalar(self.fun, method=method, **kwargs)
        assert hasattr(result, 'x')
        assert hasattr(result, 'success')
        assert hasattr(result, 'message')
        assert hasattr(result, 'fun')
        assert hasattr(result, 'nfev')
        assert hasattr(result, 'nit')

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('method', ['brent', 'bounded', 'golden'])
    def test_nan_values(self, method):
        np.random.seed(1234)
        count = [0]

        def func(x):
            count[0] += 1
            if count[0] > 4:
                return np.nan
            else:
                return x ** 2 + 0.1 * np.sin(x)
        bracket = (-1, 0, 1)
        bounds = (-1, 1)
        with np.errstate(invalid='ignore'), suppress_warnings() as sup:
            sup.filter(UserWarning, 'delta_grad == 0.*')
            sup.filter(RuntimeWarning, '.*does not use Hessian.*')
            sup.filter(RuntimeWarning, '.*does not use gradient.*')
            count = [0]
            kwargs = {'bounds': bounds} if method == 'bounded' else {}
            sol = optimize.minimize_scalar(func, bracket=bracket, **kwargs, method=method, options=dict(maxiter=20))
            assert_equal(sol.success, False)

    def test_minimize_scalar_defaults_gh10911(self):

        def f(x):
            return x ** 2
        res = optimize.minimize_scalar(f)
        assert_allclose(res.x, 0, atol=1e-08)
        res = optimize.minimize_scalar(f, bounds=(1, 100), options={'xatol': 1e-10})
        assert_allclose(res.x, 1)

    def test_minimize_non_finite_bounds_gh10911(self):
        msg = 'Optimization bounds must be finite scalars.'
        with pytest.raises(ValueError, match=msg):
            optimize.minimize_scalar(np.sin, bounds=(1, np.inf))
        with pytest.raises(ValueError, match=msg):
            optimize.minimize_scalar(np.sin, bounds=(np.nan, 1))

    @pytest.mark.parametrize('method', ['brent', 'golden'])
    def test_minimize_unbounded_method_with_bounds_gh10911(self, method):
        msg = 'Use of `bounds` is incompatible with...'
        with pytest.raises(ValueError, match=msg):
            optimize.minimize_scalar(np.sin, method=method, bounds=(1, 2))

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize('method', MINIMIZE_SCALAR_METHODS)
    @pytest.mark.parametrize('tol', [1, 1e-06])
    @pytest.mark.parametrize('fshape', [(), (1,), (1, 1)])
    def test_minimize_scalar_dimensionality_gh16196(self, method, tol, fshape):

        def f(x):
            return np.array(x ** 4).reshape(fshape)
        a, b = (-0.1, 0.2)
        kwargs = dict(bracket=(a, b)) if method != 'bounded' else dict(bounds=(a, b))
        kwargs.update(dict(method=method, tol=tol))
        res = optimize.minimize_scalar(f, **kwargs)
        assert res.x.shape == res.fun.shape == f(res.x).shape == fshape

    @pytest.mark.parametrize('method', ['bounded', 'brent', 'golden'])
    def test_minimize_scalar_warnings_gh1953(self, method):

        def f(x):
            return (x - 1) ** 2
        kwargs = {}
        kwd = 'bounds' if method == 'bounded' else 'bracket'
        kwargs[kwd] = [-2, 10]
        options = {'disp': True, 'maxiter': 3}
        with pytest.warns(optimize.OptimizeWarning, match='Maximum number'):
            optimize.minimize_scalar(f, method=method, options=options, **kwargs)
        options['disp'] = False
        optimize.minimize_scalar(f, method=method, options=options, **kwargs)