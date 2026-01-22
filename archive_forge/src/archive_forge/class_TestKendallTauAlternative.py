import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
class TestKendallTauAlternative:

    def test_kendalltau_alternative_asymptotic(self):
        x1 = [1, 2, 3, 4, 5]
        x2 = [5, 6, 7, 8, 7]
        expected = stats.kendalltau(x1, x2, alternative='two-sided')
        assert expected[0] > 0
        res = stats.kendalltau(x1, x2, alternative='less')
        assert_equal(res[0], expected[0])
        assert_allclose(res[1], 1 - expected[1] / 2)
        res = stats.kendalltau(x1, x2, alternative='greater')
        assert_equal(res[0], expected[0])
        assert_allclose(res[1], expected[1] / 2)
        x2.reverse()
        expected = stats.kendalltau(x1, x2, alternative='two-sided')
        assert expected[0] < 0
        res = stats.kendalltau(x1, x2, alternative='greater')
        assert_equal(res[0], expected[0])
        assert_allclose(res[1], 1 - expected[1] / 2)
        res = stats.kendalltau(x1, x2, alternative='less')
        assert_equal(res[0], expected[0])
        assert_allclose(res[1], expected[1] / 2)
        with pytest.raises(ValueError, match="alternative must be 'less'..."):
            stats.kendalltau(x1, x2, alternative='ekki-ekki')
    alternatives = ('less', 'two-sided', 'greater')
    p_n1 = [np.nan, np.nan, np.nan]
    p_n2 = [1, 1, 0.5]
    p_c0 = [1, 0.3333333333333, 0.1666666666667]
    p_c1 = [0.9583333333333, 0.3333333333333, 0.1666666666667]
    p_no_correlation = [0.5916666666667, 1, 0.5916666666667]
    p_no_correlationb = [0.5475694444444, 1, 0.5475694444444]
    p_n_lt_171 = [0.9624118165785, 0.1194389329806, 0.0597194664903]
    p_n_lt_171b = [0.246236925303, 0.4924738506059, 0.755634083327]
    p_n_lt_171c = [0.9847475308925, 0.03071385306533, 0.01535692653267]

    def exact_test(self, x, y, alternative, rev, stat_expected, p_expected):
        if rev:
            y = -np.asarray(y)
            stat_expected *= -1
        res = stats.kendalltau(x, y, method='exact', alternative=alternative)
        res_expected = (stat_expected, p_expected)
        assert_allclose(res, res_expected)
    case_R_n1 = list(zip(alternatives, p_n1, [False] * 3)) + list(zip(alternatives, reversed(p_n1), [True] * 3))

    @pytest.mark.parametrize('alternative, p_expected, rev', case_R_n1)
    def test_against_R_n1(self, alternative, p_expected, rev):
        x, y = ([1], [2])
        stat_expected = np.nan
        self.exact_test(x, y, alternative, rev, stat_expected, p_expected)
    case_R_n2 = list(zip(alternatives, p_n2, [False] * 3)) + list(zip(alternatives, reversed(p_n2), [True] * 3))

    @pytest.mark.parametrize('alternative, p_expected, rev', case_R_n2)
    def test_against_R_n2(self, alternative, p_expected, rev):
        x, y = ([1, 2], [3, 4])
        stat_expected = 0.9999999999999998
        self.exact_test(x, y, alternative, rev, stat_expected, p_expected)
    case_R_c0 = list(zip(alternatives, p_c0, [False] * 3)) + list(zip(alternatives, reversed(p_c0), [True] * 3))

    @pytest.mark.parametrize('alternative, p_expected, rev', case_R_c0)
    def test_against_R_c0(self, alternative, p_expected, rev):
        x, y = ([1, 2, 3], [1, 2, 3])
        stat_expected = 1
        self.exact_test(x, y, alternative, rev, stat_expected, p_expected)
    case_R_c1 = list(zip(alternatives, p_c1, [False] * 3)) + list(zip(alternatives, reversed(p_c1), [True] * 3))

    @pytest.mark.parametrize('alternative, p_expected, rev', case_R_c1)
    def test_against_R_c1(self, alternative, p_expected, rev):
        x, y = ([1, 2, 3, 4], [1, 2, 4, 3])
        stat_expected = 0.6666666666666667
        self.exact_test(x, y, alternative, rev, stat_expected, p_expected)
    case_R_no_corr = list(zip(alternatives, p_no_correlation, [False] * 3)) + list(zip(alternatives, reversed(p_no_correlation), [True] * 3))

    @pytest.mark.parametrize('alternative, p_expected, rev', case_R_no_corr)
    def test_against_R_no_correlation(self, alternative, p_expected, rev):
        x, y = ([1, 2, 3, 4, 5], [1, 5, 4, 2, 3])
        stat_expected = 0
        self.exact_test(x, y, alternative, rev, stat_expected, p_expected)
    case_no_cor_b = list(zip(alternatives, p_no_correlationb, [False] * 3)) + list(zip(alternatives, reversed(p_no_correlationb), [True] * 3))

    @pytest.mark.parametrize('alternative, p_expected, rev', case_no_cor_b)
    def test_against_R_no_correlationb(self, alternative, p_expected, rev):
        x, y = ([1, 2, 3, 4, 5, 6, 7, 8], [8, 6, 1, 3, 2, 5, 4, 7])
        stat_expected = 0
        self.exact_test(x, y, alternative, rev, stat_expected, p_expected)
    case_R_lt_171 = list(zip(alternatives, p_n_lt_171, [False] * 3)) + list(zip(alternatives, reversed(p_n_lt_171), [True] * 3))

    @pytest.mark.parametrize('alternative, p_expected, rev', case_R_lt_171)
    def test_against_R_lt_171(self, alternative, p_expected, rev):
        x = [44.4, 45.9, 41.9, 53.3, 44.7, 44.1, 50.7, 45.2, 60.1]
        y = [2.6, 3.1, 2.5, 5.0, 3.6, 4.0, 5.2, 2.8, 3.8]
        stat_expected = 0.4444444444444445
        self.exact_test(x, y, alternative, rev, stat_expected, p_expected)
    case_R_lt_171b = list(zip(alternatives, p_n_lt_171b, [False] * 3)) + list(zip(alternatives, reversed(p_n_lt_171b), [True] * 3))

    @pytest.mark.parametrize('alternative, p_expected, rev', case_R_lt_171b)
    def test_against_R_lt_171b(self, alternative, p_expected, rev):
        np.random.seed(0)
        x = np.random.rand(100)
        y = np.random.rand(100)
        stat_expected = -0.04686868686868687
        self.exact_test(x, y, alternative, rev, stat_expected, p_expected)
    case_R_lt_171c = list(zip(alternatives, p_n_lt_171c, [False] * 3)) + list(zip(alternatives, reversed(p_n_lt_171c), [True] * 3))

    @pytest.mark.parametrize('alternative, p_expected, rev', case_R_lt_171c)
    def test_against_R_lt_171c(self, alternative, p_expected, rev):
        np.random.seed(0)
        x = np.random.rand(170)
        y = np.random.rand(170)
        stat_expected = 0.1115906717716673
        self.exact_test(x, y, alternative, rev, stat_expected, p_expected)
    case_gt_171 = list(zip(alternatives, [False] * 3)) + list(zip(alternatives, [True] * 3))

    @pytest.mark.parametrize('alternative, rev', case_gt_171)
    def test_gt_171(self, alternative, rev):
        np.random.seed(0)
        x = np.random.rand(400)
        y = np.random.rand(400)
        res0 = stats.kendalltau(x, y, method='exact', alternative=alternative)
        res1 = stats.kendalltau(x, y, method='asymptotic', alternative=alternative)
        assert_equal(res0[0], res1[0])
        assert_allclose(res0[1], res1[1], rtol=0.001)

    @pytest.mark.parametrize('method', ('exact', 'asymptotic'))
    @pytest.mark.parametrize('alternative', ('two-sided', 'less', 'greater'))
    def test_nan_policy(self, method, alternative):
        x1 = [1, 2, 3, 4, 5]
        x2 = [5, 6, 7, 8, 9]
        x1nan = x1 + [np.nan]
        x2nan = x2 + [np.nan]
        res_actual = stats.kendalltau(x1nan, x2nan, method=method, alternative=alternative)
        res_expected = (np.nan, np.nan)
        assert_allclose(res_actual, res_expected)
        res_actual = stats.kendalltau(x1nan, x2nan, nan_policy='omit', method=method, alternative=alternative)
        res_expected = stats.kendalltau(x1, x2, method=method, alternative=alternative)
        assert_allclose(res_actual, res_expected)
        message = 'The input contains nan values'
        with pytest.raises(ValueError, match=message):
            stats.kendalltau(x1nan, x2nan, nan_policy='raise', method=method, alternative=alternative)
        message = 'nan_policy must be one of...'
        with pytest.raises(ValueError, match=message):
            stats.kendalltau(x1nan, x2nan, nan_policy='ekki-ekki', method=method, alternative=alternative)