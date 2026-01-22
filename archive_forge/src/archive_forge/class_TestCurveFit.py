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
class TestCurveFit:

    def setup_method(self):
        self.y = array([1.0, 3.2, 9.5, 13.7])
        self.x = array([1.0, 2.0, 3.0, 4.0])

    def test_one_argument(self):

        def func(x, a):
            return x ** a
        popt, pcov = curve_fit(func, self.x, self.y)
        assert_(len(popt) == 1)
        assert_(pcov.shape == (1, 1))
        assert_almost_equal(popt[0], 1.9149, decimal=4)
        assert_almost_equal(pcov[0, 0], 0.0016, decimal=4)
        res = curve_fit(func, self.x, self.y, full_output=1, check_finite=False)
        popt2, pcov2, infodict, errmsg, ier = res
        assert_array_almost_equal(popt, popt2)

    def test_two_argument(self):

        def func(x, a, b):
            return b * x ** a
        popt, pcov = curve_fit(func, self.x, self.y)
        assert_(len(popt) == 2)
        assert_(pcov.shape == (2, 2))
        assert_array_almost_equal(popt, [1.7989, 1.1642], decimal=4)
        assert_array_almost_equal(pcov, [[0.0852, -0.126], [-0.126, 0.1912]], decimal=4)

    def test_func_is_classmethod(self):

        class test_self:
            """This class tests if curve_fit passes the correct number of
               arguments when the model function is a class instance method.
            """

            def func(self, x, a, b):
                return b * x ** a
        test_self_inst = test_self()
        popt, pcov = curve_fit(test_self_inst.func, self.x, self.y)
        assert_(pcov.shape == (2, 2))
        assert_array_almost_equal(popt, [1.7989, 1.1642], decimal=4)
        assert_array_almost_equal(pcov, [[0.0852, -0.126], [-0.126, 0.1912]], decimal=4)

    def test_regression_2639(self):
        x = [574.142, 574.154, 574.165, 574.177, 574.188, 574.199, 574.211, 574.222, 574.234, 574.245]
        y = [859.0, 997.0, 1699.0, 2604.0, 2013.0, 1964.0, 2435.0, 1550.0, 949.0, 841.0]
        guess = [574.1861428571428, 574.2155714285715, 1302.0, 1302.0, 0.0035019999999983615, 859.0]
        good = [574.17715, 574.209188, 1741.87044, 1586.46166, 0.010068462, 857.450661]

        def f_double_gauss(x, x0, x1, A0, A1, sigma, c):
            return A0 * np.exp(-(x - x0) ** 2 / (2.0 * sigma ** 2)) + A1 * np.exp(-(x - x1) ** 2 / (2.0 * sigma ** 2)) + c
        popt, pcov = curve_fit(f_double_gauss, x, y, guess, maxfev=10000)
        assert_allclose(popt, good, rtol=1e-05)

    def test_pcov(self):
        xdata = np.array([0, 1, 2, 3, 4, 5])
        ydata = np.array([1, 1, 5, 7, 8, 12])
        sigma = np.array([1, 2, 1, 2, 1, 2])

        def f(x, a, b):
            return a * x + b
        for method in ['lm', 'trf', 'dogbox']:
            popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=sigma, method=method)
            perr_scaled = np.sqrt(np.diag(pcov))
            assert_allclose(perr_scaled, [0.20659803, 0.57204404], rtol=0.001)
            popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=3 * sigma, method=method)
            perr_scaled = np.sqrt(np.diag(pcov))
            assert_allclose(perr_scaled, [0.20659803, 0.57204404], rtol=0.001)
            popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=sigma, absolute_sigma=True, method=method)
            perr = np.sqrt(np.diag(pcov))
            assert_allclose(perr, [0.30714756, 0.85045308], rtol=0.001)
            popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=3 * sigma, absolute_sigma=True, method=method)
            perr = np.sqrt(np.diag(pcov))
            assert_allclose(perr, [3 * 0.30714756, 3 * 0.85045308], rtol=0.001)

        def f_flat(x, a, b):
            return a * x
        pcov_expected = np.array([np.inf] * 4).reshape(2, 2)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, 'Covariance of the parameters could not be estimated')
            popt, pcov = curve_fit(f_flat, xdata, ydata, p0=[2, 0], sigma=sigma)
            popt1, pcov1 = curve_fit(f, xdata[:2], ydata[:2], p0=[2, 0])
        assert_(pcov.shape == (2, 2))
        assert_array_equal(pcov, pcov_expected)
        assert_(pcov1.shape == (2, 2))
        assert_array_equal(pcov1, pcov_expected)

    def test_array_like(self):

        def f_linear(x, a, b):
            return a * x + b
        x = [1, 2, 3, 4]
        y = [3, 5, 7, 9]
        assert_allclose(curve_fit(f_linear, x, y)[0], [2, 1], atol=1e-10)

    def test_indeterminate_covariance(self):
        xdata = np.array([1, 2, 3, 4, 5, 6])
        ydata = np.array([1, 2, 3, 4, 5.5, 6])
        assert_warns(OptimizeWarning, curve_fit, lambda x, a, b: a * x, xdata, ydata)

    def test_NaN_handling(self):
        xdata = np.array([1, np.nan, 3])
        ydata = np.array([1, 2, 3])
        assert_raises(ValueError, curve_fit, lambda x, a, b: a * x + b, xdata, ydata)
        assert_raises(ValueError, curve_fit, lambda x, a, b: a * x + b, ydata, xdata)
        assert_raises(ValueError, curve_fit, lambda x, a, b: a * x + b, xdata, ydata, **{'check_finite': True})

    @staticmethod
    def _check_nan_policy(f, xdata_with_nan, xdata_without_nan, ydata_with_nan, ydata_without_nan, method):
        kwargs = {'f': f, 'xdata': xdata_with_nan, 'ydata': ydata_with_nan, 'method': method, 'check_finite': False}
        error_msg = "`nan_policy='propagate'` is not supported by this function."
        with assert_raises(ValueError, match=error_msg):
            curve_fit(**kwargs, nan_policy='propagate', maxfev=2000)
        with assert_raises(ValueError, match='The input contains nan'):
            curve_fit(**kwargs, nan_policy='raise')
        result_with_nan, _ = curve_fit(**kwargs, nan_policy='omit')
        kwargs['xdata'] = xdata_without_nan
        kwargs['ydata'] = ydata_without_nan
        result_without_nan, _ = curve_fit(**kwargs)
        assert_allclose(result_with_nan, result_without_nan)
        error_msg = "nan_policy must be one of {'None', 'raise', 'omit'}"
        with assert_raises(ValueError, match=error_msg):
            curve_fit(**kwargs, nan_policy='hi')

    @pytest.mark.parametrize('method', ['lm', 'trf', 'dogbox'])
    def test_nan_policy_1d(self, method):

        def f(x, a, b):
            return a * x + b
        xdata_with_nan = np.array([2, 3, np.nan, 4, 4, np.nan])
        ydata_with_nan = np.array([1, 2, 5, 3, np.nan, 7])
        xdata_without_nan = np.array([2, 3, 4])
        ydata_without_nan = np.array([1, 2, 3])
        self._check_nan_policy(f, xdata_with_nan, xdata_without_nan, ydata_with_nan, ydata_without_nan, method)

    @pytest.mark.parametrize('method', ['lm', 'trf', 'dogbox'])
    def test_nan_policy_2d(self, method):

        def f(x, a, b):
            x1 = x[0, :]
            x2 = x[1, :]
            return a * x1 + b + x2
        xdata_with_nan = np.array([[2, 3, np.nan, 4, 4, np.nan, 5], [2, 3, np.nan, np.nan, 4, np.nan, 7]])
        ydata_with_nan = np.array([1, 2, 5, 3, np.nan, 7, 10])
        xdata_without_nan = np.array([[2, 3, 5], [2, 3, 7]])
        ydata_without_nan = np.array([1, 2, 10])
        self._check_nan_policy(f, xdata_with_nan, xdata_without_nan, ydata_with_nan, ydata_without_nan, method)

    @pytest.mark.parametrize('n', [2, 3])
    @pytest.mark.parametrize('method', ['lm', 'trf', 'dogbox'])
    def test_nan_policy_2_3d(self, n, method):

        def f(x, a, b):
            x1 = x[..., 0, :].squeeze()
            x2 = x[..., 1, :].squeeze()
            return a * x1 + b + x2
        xdata_with_nan = np.array([[[2, 3, np.nan, 4, 4, np.nan, 5], [2, 3, np.nan, np.nan, 4, np.nan, 7]]])
        xdata_with_nan = xdata_with_nan.squeeze() if n == 2 else xdata_with_nan
        ydata_with_nan = np.array([1, 2, 5, 3, np.nan, 7, 10])
        xdata_without_nan = np.array([[[2, 3, 5], [2, 3, 7]]])
        ydata_without_nan = np.array([1, 2, 10])
        self._check_nan_policy(f, xdata_with_nan, xdata_without_nan, ydata_with_nan, ydata_without_nan, method)

    def test_empty_inputs(self):
        assert_raises(ValueError, curve_fit, lambda x, a: a * x, [], [])
        assert_raises(ValueError, curve_fit, lambda x, a: a * x, [], [], bounds=(1, 2))
        assert_raises(ValueError, curve_fit, lambda x, a: a * x, [1], [])
        assert_raises(ValueError, curve_fit, lambda x, a: a * x, [2], [], bounds=(1, 2))

    def test_function_zero_params(self):
        assert_raises(ValueError, curve_fit, lambda x: x, [1, 2], [3, 4])

    def test_None_x(self):
        popt, pcov = curve_fit(lambda _, a: a * np.arange(10), None, 2 * np.arange(10))
        assert_allclose(popt, [2.0])

    def test_method_argument(self):

        def f(x, a, b):
            return a * np.exp(-b * x)
        xdata = np.linspace(0, 1, 11)
        ydata = f(xdata, 2.0, 2.0)
        for method in ['trf', 'dogbox', 'lm', None]:
            popt, pcov = curve_fit(f, xdata, ydata, method=method)
            assert_allclose(popt, [2.0, 2.0])
        assert_raises(ValueError, curve_fit, f, xdata, ydata, method='unknown')

    def test_full_output(self):

        def f(x, a, b):
            return a * np.exp(-b * x)
        xdata = np.linspace(0, 1, 11)
        ydata = f(xdata, 2.0, 2.0)
        for method in ['trf', 'dogbox', 'lm', None]:
            popt, pcov, infodict, errmsg, ier = curve_fit(f, xdata, ydata, method=method, full_output=True)
            assert_allclose(popt, [2.0, 2.0])
            assert 'nfev' in infodict
            assert 'fvec' in infodict
            if method == 'lm' or method is None:
                assert 'fjac' in infodict
                assert 'ipvt' in infodict
                assert 'qtf' in infodict
            assert isinstance(errmsg, str)
            assert ier in (1, 2, 3, 4)

    def test_bounds(self):

        def f(x, a, b):
            return a * np.exp(-b * x)
        xdata = np.linspace(0, 1, 11)
        ydata = f(xdata, 2.0, 2.0)
        lb = [1.0, 0]
        ub = [1.5, 3.0]
        bounds = (lb, ub)
        bounds_class = Bounds(lb, ub)
        for method in [None, 'trf', 'dogbox']:
            popt, pcov = curve_fit(f, xdata, ydata, bounds=bounds, method=method)
            assert_allclose(popt[0], 1.5)
            popt_class, pcov_class = curve_fit(f, xdata, ydata, bounds=bounds_class, method=method)
            assert_allclose(popt_class, popt)
        popt, pcov = curve_fit(f, xdata, ydata, method='trf', bounds=([0.0, 0], [0.6, np.inf]))
        assert_allclose(popt[0], 0.6)
        assert_raises(ValueError, curve_fit, f, xdata, ydata, bounds=bounds, method='lm')

    def test_bounds_p0(self):

        def f(x, a):
            return np.sin(x + a)
        xdata = np.linspace(-2 * np.pi, 2 * np.pi, 40)
        ydata = np.sin(xdata)
        bounds = (-3 * np.pi, 3 * np.pi)
        for method in ['trf', 'dogbox']:
            popt_1, _ = curve_fit(f, xdata, ydata, p0=2.1 * np.pi)
            popt_2, _ = curve_fit(f, xdata, ydata, p0=2.1 * np.pi, bounds=bounds, method=method)
            assert_allclose(popt_1, popt_2)

    def test_jac(self):

        def f(x, a, b):
            return a * np.exp(-b * x)

        def jac(x, a, b):
            e = np.exp(-b * x)
            return np.vstack((e, -a * x * e)).T
        xdata = np.linspace(0, 1, 11)
        ydata = f(xdata, 2.0, 2.0)
        for method in ['trf', 'dogbox']:
            for scheme in ['2-point', '3-point', 'cs']:
                popt, pcov = curve_fit(f, xdata, ydata, jac=scheme, method=method)
                assert_allclose(popt, [2, 2])
        for method in ['lm', 'trf', 'dogbox']:
            popt, pcov = curve_fit(f, xdata, ydata, method=method, jac=jac)
            assert_allclose(popt, [2, 2])
        ydata[5] = 100
        sigma = np.ones(xdata.shape[0])
        sigma[5] = 200
        for method in ['lm', 'trf', 'dogbox']:
            popt, pcov = curve_fit(f, xdata, ydata, sigma=sigma, method=method, jac=jac)
            assert_allclose(popt, [2, 2], rtol=0.001)

    def test_maxfev_and_bounds(self):
        x = np.arange(0, 10)
        y = 2 * x
        popt1, _ = curve_fit(lambda x, p: p * x, x, y, bounds=(0, 3), maxfev=100)
        popt2, _ = curve_fit(lambda x, p: p * x, x, y, bounds=(0, 3), max_nfev=100)
        assert_allclose(popt1, 2, atol=1e-14)
        assert_allclose(popt2, 2, atol=1e-14)

    def test_curvefit_simplecovariance(self):

        def func(x, a, b):
            return a * np.exp(-b * x)

        def jac(x, a, b):
            e = np.exp(-b * x)
            return np.vstack((e, -a * x * e)).T
        np.random.seed(0)
        xdata = np.linspace(0, 4, 50)
        y = func(xdata, 2.5, 1.3)
        ydata = y + 0.2 * np.random.normal(size=len(xdata))
        sigma = np.zeros(len(xdata)) + 0.2
        covar = np.diag(sigma ** 2)
        for jac1, jac2 in [(jac, jac), (None, None)]:
            for absolute_sigma in [False, True]:
                popt1, pcov1 = curve_fit(func, xdata, ydata, sigma=sigma, jac=jac1, absolute_sigma=absolute_sigma)
                popt2, pcov2 = curve_fit(func, xdata, ydata, sigma=covar, jac=jac2, absolute_sigma=absolute_sigma)
                assert_allclose(popt1, popt2, atol=1e-14)
                assert_allclose(pcov1, pcov2, atol=1e-14)

    def test_curvefit_covariance(self):

        def funcp(x, a, b):
            rotn = np.array([[1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0], [0, 0, 1.0]])
            return rotn.dot(a * np.exp(-b * x))

        def jacp(x, a, b):
            rotn = np.array([[1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0], [0, 0, 1.0]])
            e = np.exp(-b * x)
            return rotn.dot(np.vstack((e, -a * x * e)).T)

        def func(x, a, b):
            return a * np.exp(-b * x)

        def jac(x, a, b):
            e = np.exp(-b * x)
            return np.vstack((e, -a * x * e)).T
        np.random.seed(0)
        xdata = np.arange(1, 4)
        y = func(xdata, 2.5, 1.0)
        ydata = y + 0.2 * np.random.normal(size=len(xdata))
        sigma = np.zeros(len(xdata)) + 0.2
        covar = np.diag(sigma ** 2)
        rotn = np.array([[1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0], [0, 0, 1.0]])
        ydatap = rotn.dot(ydata)
        covarp = rotn.dot(covar).dot(rotn.T)
        for jac1, jac2 in [(jac, jacp), (None, None)]:
            for absolute_sigma in [False, True]:
                popt1, pcov1 = curve_fit(func, xdata, ydata, sigma=sigma, jac=jac1, absolute_sigma=absolute_sigma)
                popt2, pcov2 = curve_fit(funcp, xdata, ydatap, sigma=covarp, jac=jac2, absolute_sigma=absolute_sigma)
                assert_allclose(popt1, popt2, rtol=1.2e-07, atol=1e-14)
                assert_allclose(pcov1, pcov2, rtol=1.2e-07, atol=1e-14)

    @pytest.mark.parametrize('absolute_sigma', [False, True])
    def test_curvefit_scalar_sigma(self, absolute_sigma):

        def func(x, a, b):
            return a * x + b
        x, y = (self.x, self.y)
        _, pcov1 = curve_fit(func, x, y, sigma=2, absolute_sigma=absolute_sigma)
        _, pcov2 = curve_fit(func, x, y, sigma=np.full_like(y, 2), absolute_sigma=absolute_sigma)
        assert np.all(pcov1 == pcov2)

    def test_dtypes(self):
        x = np.arange(-3, 5)
        y = 1.5 * x + 3.0 + 0.5 * np.sin(x)

        def func(x, a, b):
            return a * x + b
        for method in ['lm', 'trf', 'dogbox']:
            for dtx in [np.float32, np.float64]:
                for dty in [np.float32, np.float64]:
                    x = x.astype(dtx)
                    y = y.astype(dty)
                with warnings.catch_warnings():
                    warnings.simplefilter('error', OptimizeWarning)
                    p, cov = curve_fit(func, x, y, method=method)
                    assert np.isfinite(cov).all()
                    assert not np.allclose(p, 1)

    def test_dtypes2(self):

        def hyperbola(x, s_1, s_2, o_x, o_y, c):
            b_2 = (s_1 + s_2) / 2
            b_1 = (s_2 - s_1) / 2
            return o_y + b_1 * (x - o_x) + b_2 * np.sqrt((x - o_x) ** 2 + c ** 2 / 4)
        min_fit = np.array([-3.0, 0.0, -2.0, -10.0, 0.0])
        max_fit = np.array([0.0, 3.0, 3.0, 0.0, 10.0])
        guess = np.array([-2.5 / 3.0, 4 / 3.0, 1.0, -4.0, 0.5])
        params = [-2, 0.4, -1, -5, 9.5]
        xdata = np.array([-32, -16, -8, 4, 4, 8, 16, 32])
        ydata = hyperbola(xdata, *params)
        popt_64, _ = curve_fit(f=hyperbola, xdata=xdata, ydata=ydata, p0=guess, bounds=(min_fit, max_fit))
        xdata = xdata.astype(np.float32)
        ydata = hyperbola(xdata, *params)
        popt_32, _ = curve_fit(f=hyperbola, xdata=xdata, ydata=ydata, p0=guess, bounds=(min_fit, max_fit))
        assert_allclose(popt_32, popt_64, atol=2e-05)

    def test_broadcast_y(self):
        xdata = np.arange(10)
        target = 4.7 * xdata ** 2 + 3.5 * xdata + np.random.rand(len(xdata))

        def fit_func(x, a, b):
            return a * x ** 2 + b * x - target
        for method in ['lm', 'trf', 'dogbox']:
            popt0, pcov0 = curve_fit(fit_func, xdata=xdata, ydata=np.zeros_like(xdata), method=method)
            popt1, pcov1 = curve_fit(fit_func, xdata=xdata, ydata=0, method=method)
            assert_allclose(pcov0, pcov1)

    def test_args_in_kwargs(self):

        def func(x, a, b):
            return a * x + b
        with assert_raises(ValueError):
            curve_fit(func, xdata=[1, 2, 3, 4], ydata=[5, 9, 13, 17], p0=[1], args=(1,))

    def test_data_point_number_validation(self):

        def func(x, a, b, c, d, e):
            return a * np.exp(-b * x) + c + d + e
        with assert_raises(TypeError, match='The number of func parameters='):
            curve_fit(func, xdata=[1, 2, 3, 4], ydata=[5, 9, 13, 17])

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_gh4555(self):

        def f(x, a, b, c, d, e):
            return a * np.log(x + 1 + b) + c * np.log(x + 1 + d) + e
        rng = np.random.default_rng(408113519974467917)
        n = 100
        x = np.arange(n)
        y = np.linspace(2, 7, n) + rng.random(n)
        p, cov = optimize.curve_fit(f, x, y, maxfev=100000)
        assert np.all(np.diag(cov) > 0)
        eigs = linalg.eigh(cov)[0]
        assert np.all(eigs > -0.01)
        assert_allclose(cov, cov.T)

    def test_gh4555b(self):
        rng = np.random.default_rng(408113519974467917)

        def func(x, a, b, c):
            return a * np.exp(-b * x) + c
        xdata = np.linspace(0, 4, 50)
        y = func(xdata, 2.5, 1.3, 0.5)
        y_noise = 0.2 * rng.normal(size=xdata.size)
        ydata = y + y_noise
        _, res = curve_fit(func, xdata, ydata)
        ref = [[+0.0158972536486215, 0.0069207183284242, -0.0007474400714749], [+0.0069207183284242, 0.0205057958128679, +0.0053997711275403], [-0.0007474400714749, 0.0053997711275403, +0.0027833930320877]]
        assert_allclose(res, ref, 2e-07)

    def test_gh13670(self):
        rng = np.random.default_rng(8250058582555444926)
        x = np.linspace(0, 3, 101)
        y = 2 * x + 1 + rng.normal(size=101) * 0.5

        def line(x, *p):
            assert not np.all(line.last_p == p)
            line.last_p = p
            return x * p[0] + p[1]

        def jac(x, *p):
            assert not np.all(jac.last_p == p)
            jac.last_p = p
            return np.array([x, np.ones_like(x)]).T
        line.last_p = None
        jac.last_p = None
        p0 = np.array([1.0, 5.0])
        curve_fit(line, x, y, p0, method='lm', jac=jac)