import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
class TestNewton(TestScalarRootFinders):

    def test_newton_collections(self):
        known_fail = ['aps.13.00']
        known_fail += ['aps.12.05', 'aps.12.17']
        for collection in ['aps', 'complex']:
            self.run_collection(collection, zeros.newton, 'newton', smoothness=2, known_fail=known_fail)

    def test_halley_collections(self):
        known_fail = ['aps.12.06', 'aps.12.07', 'aps.12.08', 'aps.12.09', 'aps.12.10', 'aps.12.11', 'aps.12.12', 'aps.12.13', 'aps.12.14', 'aps.12.15', 'aps.12.16', 'aps.12.17', 'aps.12.18', 'aps.13.00']
        for collection in ['aps', 'complex']:
            self.run_collection(collection, zeros.newton, 'halley', smoothness=2, known_fail=known_fail)

    def test_newton(self):
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            x = zeros.newton(f, 3, tol=1e-06)
            assert_allclose(f(x), 0, atol=1e-06)
            x = zeros.newton(f, 3, x1=5, tol=1e-06)
            assert_allclose(f(x), 0, atol=1e-06)
            x = zeros.newton(f, 3, fprime=f_1, tol=1e-06)
            assert_allclose(f(x), 0, atol=1e-06)
            x = zeros.newton(f, 3, fprime=f_1, fprime2=f_2, tol=1e-06)
            assert_allclose(f(x), 0, atol=1e-06)

    def test_newton_by_name(self):
        """Invoke newton through root_scalar()"""
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='newton', x0=3, fprime=f_1, xtol=1e-06)
            assert_allclose(f(r.root), 0, atol=1e-06)
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='newton', x0=3, xtol=1e-06)
            assert_allclose(f(r.root), 0, atol=1e-06)

    def test_secant_by_name(self):
        """Invoke secant through root_scalar()"""
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='secant', x0=3, x1=2, xtol=1e-06)
            assert_allclose(f(r.root), 0, atol=1e-06)
            r = root_scalar(f, method='secant', x0=3, x1=5, xtol=1e-06)
            assert_allclose(f(r.root), 0, atol=1e-06)
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='secant', x0=3, xtol=1e-06)
            assert_allclose(f(r.root), 0, atol=1e-06)

    def test_halley_by_name(self):
        """Invoke halley through root_scalar()"""
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='halley', x0=3, fprime=f_1, fprime2=f_2, xtol=1e-06)
            assert_allclose(f(r.root), 0, atol=1e-06)

    def test_root_scalar_fail(self):
        message = 'fprime2 must be specified for halley'
        with pytest.raises(ValueError, match=message):
            root_scalar(f1, method='halley', fprime=f1_1, x0=3, xtol=1e-06)
        message = 'fprime must be specified for halley'
        with pytest.raises(ValueError, match=message):
            root_scalar(f1, method='halley', fprime2=f1_2, x0=3, xtol=1e-06)

    def test_array_newton(self):
        """test newton with array"""

        def f1(x, *a):
            b = a[0] + x * a[3]
            return a[1] - a[2] * (np.exp(b / a[5]) - 1.0) - b / a[4] - x

        def f1_1(x, *a):
            b = a[3] / a[5]
            return -a[2] * np.exp(a[0] / a[5] + x * b) * b - a[3] / a[4] - 1

        def f1_2(x, *a):
            b = a[3] / a[5]
            return -a[2] * np.exp(a[0] / a[5] + x * b) * b ** 2
        a0 = np.array([5.32725221, 5.48673747, 5.49539973, 5.36387202, 4.80237316, 1.43764452, 5.23063958, 5.46094772, 5.50512718, 5.4204629])
        a1 = (np.sin(range(10)) + 1.0) * 7.0
        args = (a0, a1, 1e-09, 0.004, 10, 0.27456)
        x0 = [7.0] * 10
        x = zeros.newton(f1, x0, f1_1, args)
        x_expected = (6.17264965, 11.7702805, 12.2219954, 7.11017681, 1.18151293, 0.143707955, 4.31928228, 10.5419107, 12.755249, 8.91225749)
        assert_allclose(x, x_expected)
        x = zeros.newton(f1, x0, f1_1, args, fprime2=f1_2)
        assert_allclose(x, x_expected)
        x = zeros.newton(f1, x0, args=args)
        assert_allclose(x, x_expected)

    def test_array_newton_complex(self):

        def f(x):
            return x + 1 + 1j

        def fprime(x):
            return 1.0
        t = np.full(4, 1j)
        x = zeros.newton(f, t, fprime=fprime)
        assert_allclose(f(x), 0.0)
        t = np.ones(4)
        x = zeros.newton(f, t, fprime=fprime)
        assert_allclose(f(x), 0.0)
        x = zeros.newton(f, t)
        assert_allclose(f(x), 0.0)

    def test_array_secant_active_zero_der(self):
        """test secant doesn't continue to iterate zero derivatives"""
        x = zeros.newton(lambda x, *a: x * x - a[0], x0=[4.123, 5], args=[np.array([17, 25])])
        assert_allclose(x, (4.123105625617661, 5.0))

    def test_array_newton_integers(self):
        x = zeros.newton(lambda y, z: z - y ** 2, [4.0] * 2, args=([15.0, 17.0],))
        assert_allclose(x, (3.872983346207417, 4.123105625617661))
        x = zeros.newton(lambda y, z: z - y ** 2, [4] * 2, args=([15, 17],))
        assert_allclose(x, (3.872983346207417, 4.123105625617661))

    def test_array_newton_zero_der_failures(self):
        assert_warns(RuntimeWarning, zeros.newton, lambda y: y ** 2 - 2, [0.0, 0.0], lambda y: 2 * y)
        with pytest.warns(RuntimeWarning):
            results = zeros.newton(lambda y: y ** 2 - 2, [0.0, 0.0], lambda y: 2 * y, full_output=True)
            assert_allclose(results.root, 0)
            assert results.zero_der.all()
            assert not results.converged.any()

    def test_newton_combined(self):

        def f1(x):
            return x ** 2 - 2 * x - 1

        def f1_1(x):
            return 2 * x - 2

        def f1_2(x):
            return 2.0 + 0 * x

        def f1_and_p_and_pp(x):
            return (x ** 2 - 2 * x - 1, 2 * x - 2, 2.0)
        sol0 = root_scalar(f1, method='newton', x0=3, fprime=f1_1)
        sol = root_scalar(f1_and_p_and_pp, method='newton', x0=3, fprime=True)
        assert_allclose(sol0.root, sol.root, atol=1e-08)
        assert_equal(2 * sol.function_calls, sol0.function_calls)
        sol0 = root_scalar(f1, method='halley', x0=3, fprime=f1_1, fprime2=f1_2)
        sol = root_scalar(f1_and_p_and_pp, method='halley', x0=3, fprime2=True)
        assert_allclose(sol0.root, sol.root, atol=1e-08)
        assert_equal(3 * sol.function_calls, sol0.function_calls)

    def test_newton_full_output(self):
        x0 = 3
        expected_counts = [(6, 7), (5, 10), (3, 9)]
        for derivs in range(3):
            kwargs = {'tol': 1e-06, 'full_output': True}
            for k, v in [['fprime', f1_1], ['fprime2', f1_2]][:derivs]:
                kwargs[k] = v
            x, r = zeros.newton(f1, x0, disp=False, **kwargs)
            assert_(r.converged)
            assert_equal(x, r.root)
            assert_equal((r.iterations, r.function_calls), expected_counts[derivs])
            if derivs == 0:
                assert r.function_calls <= r.iterations + 1
            else:
                assert_equal(r.function_calls, (derivs + 1) * r.iterations)
            iters = r.iterations - 1
            x, r = zeros.newton(f1, x0, maxiter=iters, disp=False, **kwargs)
            assert_(not r.converged)
            assert_equal(x, r.root)
            assert_equal(r.iterations, iters)
            if derivs == 1:
                msg = 'Failed to converge after %d iterations, value is .*' % iters
                with pytest.raises(RuntimeError, match=msg):
                    x, r = zeros.newton(f1, x0, maxiter=iters, disp=True, **kwargs)

    def test_deriv_zero_warning(self):

        def func(x):
            return x ** 2 - 2.0

        def dfunc(x):
            return 2 * x
        assert_warns(RuntimeWarning, zeros.newton, func, 0.0, dfunc, disp=False)
        with pytest.raises(RuntimeError, match='Derivative was zero'):
            zeros.newton(func, 0.0, dfunc)

    def test_newton_does_not_modify_x0(self):
        x0 = np.array([0.1, 3])
        x0_copy = x0.copy()
        newton(np.sin, x0, np.cos)
        assert_array_equal(x0, x0_copy)

    def test_gh17570_defaults(self):
        res_newton_default = root_scalar(f1, method='newton', x0=3, xtol=1e-06)
        res_secant_default = root_scalar(f1, method='secant', x0=3, x1=2, xtol=1e-06)
        res_secant = newton(f1, x0=3, x1=2, tol=1e-06, full_output=True)[1]
        assert_allclose(f1(res_newton_default.root), 0, atol=1e-06)
        assert res_newton_default.root.shape == tuple()
        assert_allclose(f1(res_secant_default.root), 0, atol=1e-06)
        assert res_secant_default.root.shape == tuple()
        assert_allclose(f1(res_secant.root), 0, atol=1e-06)
        assert res_secant.root.shape == tuple()
        assert res_secant_default.root == res_secant.root != res_newton_default.iterations
        assert res_secant_default.iterations == res_secant_default.function_calls - 1 == res_secant.iterations != res_newton_default.iterations == res_newton_default.function_calls / 2

    @pytest.mark.parametrize('kwargs', [dict(), {'method': 'newton'}])
    def test_args_gh19090(self, kwargs):

        def f(x, a, b):
            assert a == 3
            assert b == 1
            return x ** a - b
        res = optimize.root_scalar(f, x0=3, args=(3, 1), **kwargs)
        assert res.converged
        assert_allclose(res.root, 1)

    @pytest.mark.parametrize('method', ['secant', 'newton'])
    def test_int_x0_gh19280(self, method):

        def f(x):
            return x ** (-2) - 2
        res = optimize.root_scalar(f, x0=1, method=method)
        assert res.converged
        assert_allclose(abs(res.root), 2 ** (-0.5))
        assert res.root.dtype == np.dtype(np.float64)