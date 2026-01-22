import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
class TestWilcoxon:

    def test_wilcoxon_bad_arg(self):
        assert_raises(ValueError, stats.wilcoxon, [1], [1, 2])
        assert_raises(ValueError, stats.wilcoxon, [1, 2], [1, 2], 'dummy')
        assert_raises(ValueError, stats.wilcoxon, [1, 2], [1, 2], alternative='dummy')
        assert_raises(ValueError, stats.wilcoxon, [1] * 10, mode='xyz')

    def test_zero_diff(self):
        x = np.arange(20)
        assert_raises(ValueError, stats.wilcoxon, x, x, 'wilcox', mode='approx')
        assert_raises(ValueError, stats.wilcoxon, x, x, 'pratt', mode='approx')
        assert_equal(stats.wilcoxon(x, x, 'zsplit', mode='approx'), (20 * 21 / 4, 1.0))

    def test_pratt(self):
        x = [1, 2, 3, 4]
        y = [1, 2, 3, 5]
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message='Sample size too small')
            res = stats.wilcoxon(x, y, zero_method='pratt', mode='approx')
        assert_allclose(res, (0.0, 0.31731050786291415))

    def test_wilcoxon_arg_type(self):
        arr = [1, 2, 3, 0, -1, 3, 1, 2, 1, 1, 2]
        _ = stats.wilcoxon(arr, zero_method='pratt', mode='approx')
        _ = stats.wilcoxon(arr, zero_method='zsplit', mode='approx')
        _ = stats.wilcoxon(arr, zero_method='wilcox', mode='approx')

    def test_accuracy_wilcoxon(self):
        freq = [1, 4, 16, 15, 8, 4, 5, 1, 2]
        nums = range(-4, 5)
        x = np.concatenate([[u] * v for u, v in zip(nums, freq)])
        y = np.zeros(x.size)
        T, p = stats.wilcoxon(x, y, 'pratt', mode='approx')
        assert_allclose(T, 423)
        assert_allclose(p, 0.0031724568006762576)
        T, p = stats.wilcoxon(x, y, 'zsplit', mode='approx')
        assert_allclose(T, 441)
        assert_allclose(p, 0.0032145343172473055)
        T, p = stats.wilcoxon(x, y, 'wilcox', mode='approx')
        assert_allclose(T, 327)
        assert_allclose(p, 0.00641346115861)
        x = np.array([120, 114, 181, 188, 180, 146, 121, 191, 132, 113, 127, 112])
        y = np.array([133, 143, 119, 189, 112, 199, 198, 113, 115, 121, 142, 187])
        T, p = stats.wilcoxon(x, y, correction=False, mode='approx')
        assert_equal(T, 34)
        assert_allclose(p, 0.6948866, rtol=1e-06)
        T, p = stats.wilcoxon(x, y, correction=True, mode='approx')
        assert_equal(T, 34)
        assert_allclose(p, 0.7240817, rtol=1e-06)

    def test_wilcoxon_result_attributes(self):
        x = np.array([120, 114, 181, 188, 180, 146, 121, 191, 132, 113, 127, 112])
        y = np.array([133, 143, 119, 189, 112, 199, 198, 113, 115, 121, 142, 187])
        res = stats.wilcoxon(x, y, correction=False, mode='approx')
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

    def test_wilcoxon_has_zstatistic(self):
        rng = np.random.default_rng(89426135444)
        x, y = (rng.random(15), rng.random(15))
        res = stats.wilcoxon(x, y, mode='approx')
        ref = stats.norm.ppf(res.pvalue / 2)
        assert_allclose(res.zstatistic, ref)
        res = stats.wilcoxon(x, y, mode='exact')
        assert not hasattr(res, 'zstatistic')
        res = stats.wilcoxon(x, y)
        assert not hasattr(res, 'zstatistic')

    def test_wilcoxon_tie(self):
        stat, p = stats.wilcoxon([0.1] * 10, mode='approx')
        expected_p = 0.001565402
        assert_equal(stat, 0)
        assert_allclose(p, expected_p, rtol=1e-06)
        stat, p = stats.wilcoxon([0.1] * 10, correction=True, mode='approx')
        expected_p = 0.001904195
        assert_equal(stat, 0)
        assert_allclose(p, expected_p, rtol=1e-06)

    def test_onesided(self):
        x = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135]
        y = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message='Sample size too small')
            w, p = stats.wilcoxon(x, y, alternative='less', mode='approx')
        assert_equal(w, 27)
        assert_almost_equal(p, 0.7031847, decimal=6)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message='Sample size too small')
            w, p = stats.wilcoxon(x, y, alternative='less', correction=True, mode='approx')
        assert_equal(w, 27)
        assert_almost_equal(p, 0.7233656, decimal=6)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message='Sample size too small')
            w, p = stats.wilcoxon(x, y, alternative='greater', mode='approx')
        assert_equal(w, 27)
        assert_almost_equal(p, 0.2968153, decimal=6)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message='Sample size too small')
            w, p = stats.wilcoxon(x, y, alternative='greater', correction=True, mode='approx')
        assert_equal(w, 27)
        assert_almost_equal(p, 0.3176447, decimal=6)

    def test_exact_basic(self):
        for n in range(1, 51):
            pmf1 = _get_wilcoxon_distr(n)
            pmf2 = _get_wilcoxon_distr2(n)
            assert_equal(n * (n + 1) / 2 + 1, len(pmf1))
            assert_equal(sum(pmf1), 1)
            assert_array_almost_equal(pmf1, pmf2)

    def test_exact_pval(self):
        x = np.array([1.81, 0.82, 1.56, -0.48, 0.81, 1.28, -1.04, 0.23, -0.75, 0.14])
        y = np.array([0.71, 0.65, -0.2, 0.85, -1.1, -0.45, -0.84, -0.24, -0.68, -0.76])
        _, p = stats.wilcoxon(x, y, alternative='two-sided', mode='exact')
        assert_almost_equal(p, 0.1054688, decimal=6)
        _, p = stats.wilcoxon(x, y, alternative='less', mode='exact')
        assert_almost_equal(p, 0.9580078, decimal=6)
        _, p = stats.wilcoxon(x, y, alternative='greater', mode='exact')
        assert_almost_equal(p, 0.05273438, decimal=6)
        x = np.arange(0, 20) + 0.5
        y = np.arange(20, 0, -1)
        _, p = stats.wilcoxon(x, y, alternative='two-sided', mode='exact')
        assert_almost_equal(p, 0.8694878, decimal=6)
        _, p = stats.wilcoxon(x, y, alternative='less', mode='exact')
        assert_almost_equal(p, 0.4347439, decimal=6)
        _, p = stats.wilcoxon(x, y, alternative='greater', mode='exact')
        assert_almost_equal(p, 0.5795889, decimal=6)

    @pytest.mark.parametrize('x', [[-1, -2, 3], [-1, 2, -3, -4, 5], [-1, -2, 3, -4, -5, -6, 7, 8]])
    def test_exact_p_1(self, x):
        w, p = stats.wilcoxon(x)
        x = np.array(x)
        wtrue = x[x > 0].sum()
        assert_equal(w, wtrue)
        assert_equal(p, 1)

    def test_auto(self):
        x = np.arange(0, 25) + 0.5
        y = np.arange(25, 0, -1)
        assert_equal(stats.wilcoxon(x, y), stats.wilcoxon(x, y, mode='exact'))
        d = np.arange(0, 13)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message='Exact p-value calculation')
            w, p = stats.wilcoxon(d)
        assert_equal(stats.wilcoxon(d, mode='approx'), (w, p))
        d = np.arange(1, 52)
        assert_equal(stats.wilcoxon(d), stats.wilcoxon(d, mode='approx'))