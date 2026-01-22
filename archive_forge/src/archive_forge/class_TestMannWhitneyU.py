from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
class TestMannWhitneyU:

    def setup_method(self):
        _mwu_state._recursive = True

    def test_input_validation(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        with assert_raises(ValueError, match='`x` and `y` must be of nonzero'):
            mannwhitneyu([], y)
        with assert_raises(ValueError, match='`x` and `y` must be of nonzero'):
            mannwhitneyu(x, [])
        with assert_raises(ValueError, match='`use_continuity` must be one'):
            mannwhitneyu(x, y, use_continuity='ekki')
        with assert_raises(ValueError, match='`alternative` must be one of'):
            mannwhitneyu(x, y, alternative='ekki')
        with assert_raises(ValueError, match='`axis` must be an integer'):
            mannwhitneyu(x, y, axis=1.5)
        with assert_raises(ValueError, match='`method` must be one of'):
            mannwhitneyu(x, y, method='ekki')

    def test_auto(self):
        np.random.seed(1)
        n = 8
        x = np.random.rand(n - 1)
        y = np.random.rand(n - 1)
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue == exact.pvalue
        assert auto.pvalue != asymptotic.pvalue
        x = np.random.rand(n - 1)
        y = np.random.rand(n + 1)
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue == exact.pvalue
        assert auto.pvalue != asymptotic.pvalue
        auto = mannwhitneyu(y, x)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue == exact.pvalue
        assert auto.pvalue != asymptotic.pvalue
        x = np.random.rand(n + 1)
        y = np.random.rand(n + 1)
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue != exact.pvalue
        assert auto.pvalue == asymptotic.pvalue
        x = np.random.rand(n - 1)
        y = np.random.rand(n - 1)
        y[3] = x[3]
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue != exact.pvalue
        assert auto.pvalue == asymptotic.pvalue
    x = [210.05211, 110.19063, 307.918612]
    y = [436.08811482466416, 416.3739732976819, 179.96975939463582, 197.8118754228619, 34.038757281225756, 138.54220550921517, 128.7769351470246, 265.9272142795185, 275.6617533155341, 592.3408339541626, 448.7317759061702, 300.61495185038905, 187.97508449019588]
    cases_basic = [[{'alternative': 'two-sided', 'method': 'asymptotic'}, (16, 0.6865041817876)], [{'alternative': 'less', 'method': 'asymptotic'}, (16, 0.3432520908938)], [{'alternative': 'greater', 'method': 'asymptotic'}, (16, 0.7047591913255)], [{'alternative': 'two-sided', 'method': 'exact'}, (16, 0.7035714285714)], [{'alternative': 'less', 'method': 'exact'}, (16, 0.3517857142857)], [{'alternative': 'greater', 'method': 'exact'}, (16, 0.6946428571429)]]

    @pytest.mark.parametrize(('kwds', 'expected'), cases_basic)
    def test_basic(self, kwds, expected):
        res = mannwhitneyu(self.x, self.y, **kwds)
        assert_allclose(res, expected)
    cases_continuity = [[{'alternative': 'two-sided', 'use_continuity': True}, (23, 0.6865041817876)], [{'alternative': 'less', 'use_continuity': True}, (23, 0.7047591913255)], [{'alternative': 'greater', 'use_continuity': True}, (23, 0.3432520908938)], [{'alternative': 'two-sided', 'use_continuity': False}, (23, 0.6377328900502)], [{'alternative': 'less', 'use_continuity': False}, (23, 0.6811335549749)], [{'alternative': 'greater', 'use_continuity': False}, (23, 0.3188664450251)]]

    @pytest.mark.parametrize(('kwds', 'expected'), cases_continuity)
    def test_continuity(self, kwds, expected):
        res = mannwhitneyu(self.y, self.x, method='asymptotic', **kwds)
        assert_allclose(res, expected)

    def test_tie_correct(self):
        x = [1, 2, 3, 4]
        y0 = np.array([1, 2, 3, 4, 5])
        dy = np.array([0, 1, 0, 1, 0]) * 0.01
        dy2 = np.array([0, 0, 1, 0, 0]) * 0.01
        y = [y0 - 0.01, y0 - dy, y0 - dy2, y0, y0 + dy2, y0 + dy, y0 + 0.01]
        res = mannwhitneyu(x, y, axis=-1, method='asymptotic')
        U_expected = [10, 9, 8.5, 8, 7.5, 7, 6]
        p_expected = [1, 0.9017048037317, 0.804080657472, 0.7086240584439, 0.6197963884941, 0.5368784563079, 0.3912672792826]
        assert_equal(res.statistic, U_expected)
        assert_allclose(res.pvalue, p_expected)
    pn3 = {1: [0.25, 0.5, 0.75], 2: [0.1, 0.2, 0.4, 0.6], 3: [0.05, 0.1, 0.2, 0.35, 0.5, 0.65]}
    pn4 = {1: [0.2, 0.4, 0.6], 2: [0.067, 0.133, 0.267, 0.4, 0.6], 3: [0.028, 0.057, 0.114, 0.2, 0.314, 0.429, 0.571], 4: [0.014, 0.029, 0.057, 0.1, 0.171, 0.243, 0.343, 0.443, 0.557]}
    pm5 = {1: [0.167, 0.333, 0.5, 0.667], 2: [0.047, 0.095, 0.19, 0.286, 0.429, 0.571], 3: [0.018, 0.036, 0.071, 0.125, 0.196, 0.286, 0.393, 0.5, 0.607], 4: [0.008, 0.016, 0.032, 0.056, 0.095, 0.143, 0.206, 0.278, 0.365, 0.452, 0.548], 5: [0.004, 0.008, 0.016, 0.028, 0.048, 0.075, 0.111, 0.155, 0.21, 0.274, 0.345, 0.421, 0.5, 0.579]}
    pm6 = {1: [0.143, 0.286, 0.428, 0.571], 2: [0.036, 0.071, 0.143, 0.214, 0.321, 0.429, 0.571], 3: [0.012, 0.024, 0.048, 0.083, 0.131, 0.19, 0.274, 0.357, 0.452, 0.548], 4: [0.005, 0.01, 0.019, 0.033, 0.057, 0.086, 0.129, 0.176, 0.238, 0.305, 0.381, 0.457, 0.543], 5: [0.002, 0.004, 0.009, 0.015, 0.026, 0.041, 0.063, 0.089, 0.123, 0.165, 0.214, 0.268, 0.331, 0.396, 0.465, 0.535], 6: [0.001, 0.002, 0.004, 0.008, 0.013, 0.021, 0.032, 0.047, 0.066, 0.09, 0.12, 0.155, 0.197, 0.242, 0.294, 0.35, 0.409, 0.469, 0.531]}

    def test_exact_distribution(self):
        p_tables = {3: self.pn3, 4: self.pn4, 5: self.pm5, 6: self.pm6}
        for n, table in p_tables.items():
            for m, p in table.items():
                u = np.arange(0, len(p))
                assert_allclose(_mwu_state.cdf(k=u, m=m, n=n), p, atol=0.001)
                u2 = np.arange(0, m * n + 1)
                assert_allclose(_mwu_state.cdf(k=u2, m=m, n=n) + _mwu_state.sf(k=u2, m=m, n=n) - _mwu_state.pmf(k=u2, m=m, n=n), 1)
                pmf = _mwu_state.pmf(k=u2, m=m, n=n)
                assert_allclose(pmf, pmf[::-1])
                pmf2 = _mwu_state.pmf(k=u2, m=n, n=m)
                assert_allclose(pmf, pmf2)

    def test_asymptotic_behavior(self):
        np.random.seed(0)
        x = np.random.rand(5)
        y = np.random.rand(5)
        res1 = mannwhitneyu(x, y, method='exact')
        res2 = mannwhitneyu(x, y, method='asymptotic')
        assert res1.statistic == res2.statistic
        assert np.abs(res1.pvalue - res2.pvalue) > 0.01
        x = np.random.rand(40)
        y = np.random.rand(40)
        res1 = mannwhitneyu(x, y, method='exact')
        res2 = mannwhitneyu(x, y, method='asymptotic')
        assert res1.statistic == res2.statistic
        assert np.abs(res1.pvalue - res2.pvalue) < 0.001

    def test_exact_U_equals_mean(self):
        res_l = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative='less', method='exact')
        res_g = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative='greater', method='exact')
        assert_equal(res_l.pvalue, res_g.pvalue)
        assert res_l.pvalue > 0.5
        res = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative='two-sided', method='exact')
        assert_equal(res, (3, 1))
    cases_scalar = [[{'alternative': 'two-sided', 'method': 'asymptotic'}, (0, 1)], [{'alternative': 'less', 'method': 'asymptotic'}, (0, 0.5)], [{'alternative': 'greater', 'method': 'asymptotic'}, (0, 0.977249868052)], [{'alternative': 'two-sided', 'method': 'exact'}, (0, 1)], [{'alternative': 'less', 'method': 'exact'}, (0, 0.5)], [{'alternative': 'greater', 'method': 'exact'}, (0, 1)]]

    @pytest.mark.parametrize(('kwds', 'result'), cases_scalar)
    def test_scalar_data(self, kwds, result):
        assert_allclose(mannwhitneyu(1, 2, **kwds), result)

    def test_equal_scalar_data(self):
        assert_equal(mannwhitneyu(1, 1, method='exact'), (0.5, 1))
        assert_equal(mannwhitneyu(1, 1, method='asymptotic'), (0.5, 1))
        assert_equal(mannwhitneyu(1, 1, method='asymptotic', use_continuity=False), (0.5, np.nan))

    @pytest.mark.parametrize('method', ['asymptotic', 'exact'])
    def test_gh_12837_11113(self, method):
        np.random.seed(0)
        axis = -3
        m, n = (7, 10)
        x = np.random.rand(m, 3, 8)
        y = np.random.rand(6, n, 1, 8) + 0.1
        res = mannwhitneyu(x, y, method=method, axis=axis)
        shape = (6, 3, 8)
        assert res.pvalue.shape == shape
        assert res.statistic.shape == shape
        x, y = (np.moveaxis(x, axis, -1), np.moveaxis(y, axis, -1))
        x = x[None, ...]
        assert x.ndim == y.ndim
        x = np.broadcast_to(x, shape + (m,))
        y = np.broadcast_to(y, shape + (n,))
        assert x.shape[:-1] == shape
        assert y.shape[:-1] == shape
        statistics = np.zeros(shape)
        pvalues = np.zeros(shape)
        for indices in product(*[range(i) for i in shape]):
            xi = x[indices]
            yi = y[indices]
            temp = mannwhitneyu(xi, yi, method=method)
            statistics[indices] = temp.statistic
            pvalues[indices] = temp.pvalue
        np.testing.assert_equal(res.pvalue, pvalues)
        np.testing.assert_equal(res.statistic, statistics)

    def test_gh_11355(self):
        x = [1, 2, 3, 4]
        y = [3, 6, 7, 8, 9, 3, 2, 1, 4, 4, 5]
        res1 = mannwhitneyu(x, y)
        y[4] = np.inf
        res2 = mannwhitneyu(x, y)
        assert_equal(res1.statistic, res2.statistic)
        assert_equal(res1.pvalue, res2.pvalue)
        y[4] = np.nan
        res3 = mannwhitneyu(x, y)
        assert_equal(res3.statistic, np.nan)
        assert_equal(res3.pvalue, np.nan)
    cases_11355 = [([1, 2, 3, 4], [3, 6, 7, 8, np.inf, 3, 2, 1, 4, 4, 5], 10, 0.1297704873477), ([1, 2, 3, 4], [3, 6, 7, 8, np.inf, np.inf, 2, 1, 4, 4, 5], 8.5, 0.08735617507695), ([1, 2, np.inf, 4], [3, 6, 7, 8, np.inf, 3, 2, 1, 4, 4, 5], 17.5, 0.5988856695752), ([1, 2, np.inf, 4], [3, 6, 7, 8, np.inf, np.inf, 2, 1, 4, 4, 5], 16, 0.4687165824462), ([1, np.inf, np.inf, 4], [3, 6, 7, 8, np.inf, np.inf, 2, 1, 4, 4, 5], 24.5, 0.7912517950119)]

    @pytest.mark.parametrize(('x', 'y', 'statistic', 'pvalue'), cases_11355)
    def test_gh_11355b(self, x, y, statistic, pvalue):
        res = mannwhitneyu(x, y, method='asymptotic')
        assert_allclose(res.statistic, statistic, atol=1e-12)
        assert_allclose(res.pvalue, pvalue, atol=1e-12)
    cases_9184 = [[True, 'less', 'asymptotic', 0.900775348204], [True, 'greater', 'asymptotic', 0.1223118025635], [True, 'two-sided', 'asymptotic', 0.244623605127], [False, 'less', 'asymptotic', 0.8896643190401], [False, 'greater', 'asymptotic', 0.1103356809599], [False, 'two-sided', 'asymptotic', 0.2206713619198], [True, 'less', 'exact', 0.8967698967699], [True, 'greater', 'exact', 0.1272061272061], [True, 'two-sided', 'exact', 0.2544122544123]]

    @pytest.mark.parametrize(('use_continuity', 'alternative', 'method', 'pvalue_exp'), cases_9184)
    def test_gh_9184(self, use_continuity, alternative, method, pvalue_exp):
        statistic_exp = 35
        x = (0.8, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46)
        y = (1.15, 0.88, 0.9, 0.74, 1.21)
        res = mannwhitneyu(x, y, use_continuity=use_continuity, alternative=alternative, method=method)
        assert_equal(res.statistic, statistic_exp)
        assert_allclose(res.pvalue, pvalue_exp)

    def test_gh_6897(self):
        with assert_raises(ValueError, match='`x` and `y` must be of nonzero'):
            mannwhitneyu([], [])

    def test_gh_4067(self):
        a = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        b = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        res = mannwhitneyu(a, b)
        assert_equal(res.statistic, np.nan)
        assert_equal(res.pvalue, np.nan)
    cases_2118 = [[[1, 2, 3], [1.5, 2.5], 'greater', (3, 0.6135850036578)], [[1, 2, 3], [1.5, 2.5], 'less', (3, 0.6135850036578)], [[1, 2, 3], [1.5, 2.5], 'two-sided', (3, 1.0)], [[1, 2, 3], [2], 'greater', (1.5, 0.681324055883)], [[1, 2, 3], [2], 'less', (1.5, 0.681324055883)], [[1, 2, 3], [2], 'two-sided', (1.5, 1)], [[1, 2], [1, 2], 'greater', (2, 0.667497228949)], [[1, 2], [1, 2], 'less', (2, 0.667497228949)], [[1, 2], [1, 2], 'two-sided', (2, 1)]]

    @pytest.mark.parametrize(['x', 'y', 'alternative', 'expected'], cases_2118)
    def test_gh_2118(self, x, y, alternative, expected):
        res = mannwhitneyu(x, y, use_continuity=True, alternative=alternative, method='asymptotic')
        assert_allclose(res, expected, rtol=1e-12)

    def teardown_method(self):
        _mwu_state._recursive = None