import pytest
import numpy as np
from numpy import cos, sin, pi
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyp_num
from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
from scipy.integrate._quadrature import _cumulative_simpson_unequal_intervals
from scipy.integrate._tanhsinh import _tanhsinh, _pair_cache
from scipy import stats, special as sc
from scipy.optimize._zeros_py import (_ECONVERGED, _ESIGNERR, _ECONVERR,  # noqa: F401
class TestQMCQuad:

    def test_input_validation(self):
        message = '`func` must be callable.'
        with pytest.raises(TypeError, match=message):
            qmc_quad('a duck', [0, 0], [1, 1])
        message = '`func` must evaluate the integrand at points...'
        with pytest.raises(ValueError, match=message):
            qmc_quad(lambda: 1, [0, 0], [1, 1])

        def func(x):
            assert x.ndim == 1
            return np.sum(x)
        message = 'Exception encountered when attempting vectorized call...'
        with pytest.warns(UserWarning, match=message):
            qmc_quad(func, [0, 0], [1, 1])
        message = '`n_points` must be an integer.'
        with pytest.raises(TypeError, match=message):
            qmc_quad(lambda x: 1, [0, 0], [1, 1], n_points=1024.5)
        message = '`n_estimates` must be an integer.'
        with pytest.raises(TypeError, match=message):
            qmc_quad(lambda x: 1, [0, 0], [1, 1], n_estimates=8.5)
        message = '`qrng` must be an instance of scipy.stats.qmc.QMCEngine.'
        with pytest.raises(TypeError, match=message):
            qmc_quad(lambda x: 1, [0, 0], [1, 1], qrng='a duck')
        message = '`qrng` must be initialized with dimensionality equal to '
        with pytest.raises(ValueError, match=message):
            qmc_quad(lambda x: 1, [0, 0], [1, 1], qrng=stats.qmc.Sobol(1))
        message = '`log` must be boolean \\(`True` or `False`\\).'
        with pytest.raises(TypeError, match=message):
            qmc_quad(lambda x: 1, [0, 0], [1, 1], log=10)

    def basic_test(self, n_points=2 ** 8, n_estimates=8, signs=np.ones(2)):
        ndim = 2
        mean = np.zeros(ndim)
        cov = np.eye(ndim)

        def func(x):
            return stats.multivariate_normal.pdf(x.T, mean, cov)
        rng = np.random.default_rng(2879434385674690281)
        qrng = stats.qmc.Sobol(ndim, seed=rng)
        a = np.zeros(ndim)
        b = np.ones(ndim) * signs
        res = qmc_quad(func, a, b, n_points=n_points, n_estimates=n_estimates, qrng=qrng)
        ref = stats.multivariate_normal.cdf(b, mean, cov, lower_limit=a)
        atol = sc.stdtrit(n_estimates - 1, 0.995) * res.standard_error
        assert_allclose(res.integral, ref, atol=atol)
        assert np.prod(signs) * res.integral > 0
        rng = np.random.default_rng(2879434385674690281)
        qrng = stats.qmc.Sobol(ndim, seed=rng)
        logres = qmc_quad(lambda *args: np.log(func(*args)), a, b, n_points=n_points, n_estimates=n_estimates, log=True, qrng=qrng)
        assert_allclose(np.exp(logres.integral), res.integral, rtol=1e-14)
        assert np.imag(logres.integral) == (np.pi if np.prod(signs) < 0 else 0)
        assert_allclose(np.exp(logres.standard_error), res.standard_error, rtol=1e-14, atol=1e-16)

    @pytest.mark.parametrize('n_points', [2 ** 8, 2 ** 12])
    @pytest.mark.parametrize('n_estimates', [8, 16])
    def test_basic(self, n_points, n_estimates):
        self.basic_test(n_points, n_estimates)

    @pytest.mark.parametrize('signs', [[1, 1], [-1, -1], [-1, 1], [1, -1]])
    def test_sign(self, signs):
        self.basic_test(signs=signs)

    @pytest.mark.parametrize('log', [False, True])
    def test_zero(self, log):
        message = 'A lower limit was equal to an upper limit, so'
        with pytest.warns(UserWarning, match=message):
            res = qmc_quad(lambda x: 1, [0, 0], [0, 1], log=log)
        assert res.integral == (-np.inf if log else 0)
        assert res.standard_error == 0

    def test_flexible_input(self):

        def func(x):
            return stats.norm.pdf(x, scale=2)
        res = qmc_quad(func, 0, 1)
        ref = stats.norm.cdf(1, scale=2) - stats.norm.cdf(0, scale=2)
        assert_allclose(res.integral, ref, 0.01)