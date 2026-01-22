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
class TestAndersonKSamp:

    def test_example1a(self):
        t1 = [38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0]
        t2 = np.array([39.2, 39.3, 39.7, 41.4, 41.8, 42.9, 43.3, 45.8])
        t3 = np.array([34.0, 35.0, 39.0, 40.0, 43.0, 43.0, 44.0, 45.0])
        t4 = np.array([34.0, 34.8, 34.8, 35.4, 37.2, 37.8, 41.2, 42.8])
        Tk, tm, p = stats.anderson_ksamp((t1, t2, t3, t4), midrank=False)
        assert_almost_equal(Tk, 4.449, 3)
        assert_array_almost_equal([0.4985, 1.3237, 1.9158, 2.493, 3.2459], tm[0:5], 4)
        assert_allclose(p, 0.0021, atol=0.00025)

    def test_example1b(self):
        t1 = np.array([38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0])
        t2 = np.array([39.2, 39.3, 39.7, 41.4, 41.8, 42.9, 43.3, 45.8])
        t3 = np.array([34.0, 35.0, 39.0, 40.0, 43.0, 43.0, 44.0, 45.0])
        t4 = np.array([34.0, 34.8, 34.8, 35.4, 37.2, 37.8, 41.2, 42.8])
        Tk, tm, p = stats.anderson_ksamp((t1, t2, t3, t4), midrank=True)
        assert_almost_equal(Tk, 4.48, 3)
        assert_array_almost_equal([0.4985, 1.3237, 1.9158, 2.493, 3.2459], tm[0:5], 4)
        assert_allclose(p, 0.002, atol=0.00025)

    @pytest.mark.slow
    def test_example2a(self):
        t1 = [194, 15, 41, 29, 33, 181]
        t2 = [413, 14, 58, 37, 100, 65, 9, 169, 447, 184, 36, 201, 118]
        t3 = [34, 31, 18, 18, 67, 57, 62, 7, 22, 34]
        t4 = [90, 10, 60, 186, 61, 49, 14, 24, 56, 20, 79, 84, 44, 59, 29, 118, 25, 156, 310, 76, 26, 44, 23, 62]
        t5 = [130, 208, 70, 101, 208]
        t6 = [74, 57, 48, 29, 502, 12, 70, 21, 29, 386, 59, 27]
        t7 = [55, 320, 56, 104, 220, 239, 47, 246, 176, 182, 33]
        t8 = [23, 261, 87, 7, 120, 14, 62, 47, 225, 71, 246, 21, 42, 20, 5, 12, 120, 11, 3, 14, 71, 11, 14, 11, 16, 90, 1, 16, 52, 95]
        t9 = [97, 51, 11, 4, 141, 18, 142, 68, 77, 80, 1, 16, 106, 206, 82, 54, 31, 216, 46, 111, 39, 63, 18, 191, 18, 163, 24]
        t10 = [50, 44, 102, 72, 22, 39, 3, 15, 197, 188, 79, 88, 46, 5, 5, 36, 22, 139, 210, 97, 30, 23, 13, 14]
        t11 = [359, 9, 12, 270, 603, 3, 104, 2, 438]
        t12 = [50, 254, 5, 283, 35, 12]
        t13 = [487, 18, 100, 7, 98, 5, 85, 91, 43, 230, 3, 130]
        t14 = [102, 209, 14, 57, 54, 32, 67, 59, 134, 152, 27, 14, 230, 66, 61, 34]
        samples = (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14)
        Tk, tm, p = stats.anderson_ksamp(samples, midrank=False)
        assert_almost_equal(Tk, 3.288, 3)
        assert_array_almost_equal([0.599, 1.3269, 1.8052, 2.2486, 2.8009], tm[0:5], 4)
        assert_allclose(p, 0.0041, atol=0.00025)
        rng = np.random.default_rng(6989860141921615054)
        method = stats.PermutationMethod(n_resamples=9999, random_state=rng)
        res = stats.anderson_ksamp(samples, midrank=False, method=method)
        assert_array_equal(res.statistic, Tk)
        assert_array_equal(res.critical_values, tm)
        assert_allclose(res.pvalue, p, atol=0.0006)

    def test_example2b(self):
        t1 = [194, 15, 41, 29, 33, 181]
        t2 = [413, 14, 58, 37, 100, 65, 9, 169, 447, 184, 36, 201, 118]
        t3 = [34, 31, 18, 18, 67, 57, 62, 7, 22, 34]
        t4 = [90, 10, 60, 186, 61, 49, 14, 24, 56, 20, 79, 84, 44, 59, 29, 118, 25, 156, 310, 76, 26, 44, 23, 62]
        t5 = [130, 208, 70, 101, 208]
        t6 = [74, 57, 48, 29, 502, 12, 70, 21, 29, 386, 59, 27]
        t7 = [55, 320, 56, 104, 220, 239, 47, 246, 176, 182, 33]
        t8 = [23, 261, 87, 7, 120, 14, 62, 47, 225, 71, 246, 21, 42, 20, 5, 12, 120, 11, 3, 14, 71, 11, 14, 11, 16, 90, 1, 16, 52, 95]
        t9 = [97, 51, 11, 4, 141, 18, 142, 68, 77, 80, 1, 16, 106, 206, 82, 54, 31, 216, 46, 111, 39, 63, 18, 191, 18, 163, 24]
        t10 = [50, 44, 102, 72, 22, 39, 3, 15, 197, 188, 79, 88, 46, 5, 5, 36, 22, 139, 210, 97, 30, 23, 13, 14]
        t11 = [359, 9, 12, 270, 603, 3, 104, 2, 438]
        t12 = [50, 254, 5, 283, 35, 12]
        t13 = [487, 18, 100, 7, 98, 5, 85, 91, 43, 230, 3, 130]
        t14 = [102, 209, 14, 57, 54, 32, 67, 59, 134, 152, 27, 14, 230, 66, 61, 34]
        Tk, tm, p = stats.anderson_ksamp((t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14), midrank=True)
        assert_almost_equal(Tk, 3.294, 3)
        assert_array_almost_equal([0.599, 1.3269, 1.8052, 2.2486, 2.8009], tm[0:5], 4)
        assert_allclose(p, 0.0041, atol=0.00025)

    def test_R_kSamples(self):
        x1 = np.linspace(1, 100, 100)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message='p-value floored')
            s, _, p = stats.anderson_ksamp([x1, x1 + 40.5], midrank=False)
        assert_almost_equal(s, 41.105, 3)
        assert_equal(p, 0.001)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message='p-value floored')
            s, _, p = stats.anderson_ksamp([x1, x1 + 40.5])
        assert_almost_equal(s, 41.235, 3)
        assert_equal(p, 0.001)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message='p-value capped')
            s, _, p = stats.anderson_ksamp([x1, x1 + 0.5], midrank=False)
        assert_almost_equal(s, -1.2824, 4)
        assert_equal(p, 0.25)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message='p-value capped')
            s, _, p = stats.anderson_ksamp([x1, x1 + 0.5])
        assert_almost_equal(s, -1.2944, 4)
        assert_equal(p, 0.25)
        s, _, p = stats.anderson_ksamp([x1, x1 + 7.5], midrank=False)
        assert_almost_equal(s, 1.4923, 4)
        assert_allclose(p, 0.0775, atol=0.005, rtol=0)
        s, _, p = stats.anderson_ksamp([x1, x1 + 6])
        assert_almost_equal(s, 0.6389, 4)
        assert_allclose(p, 0.1798, atol=0.005, rtol=0)
        s, _, p = stats.anderson_ksamp([x1, x1 + 11.5], midrank=False)
        assert_almost_equal(s, 4.5042, 4)
        assert_allclose(p, 0.00545, atol=0.0005, rtol=0)
        s, _, p = stats.anderson_ksamp([x1, x1 + 13.5], midrank=False)
        assert_almost_equal(s, 6.2982, 4)
        assert_allclose(p, 0.00118, atol=0.0001, rtol=0)

    def test_not_enough_samples(self):
        assert_raises(ValueError, stats.anderson_ksamp, np.ones(5))

    def test_no_distinct_observations(self):
        assert_raises(ValueError, stats.anderson_ksamp, (np.ones(5), np.ones(5)))

    def test_empty_sample(self):
        assert_raises(ValueError, stats.anderson_ksamp, (np.ones(5), []))

    def test_result_attributes(self):
        t1 = [38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0]
        t2 = np.array([39.2, 39.3, 39.7, 41.4, 41.8, 42.9, 43.3, 45.8])
        res = stats.anderson_ksamp((t1, t2), midrank=False)
        attributes = ('statistic', 'critical_values', 'significance_level')
        check_named_results(res, attributes)
        assert_equal(res.significance_level, res.pvalue)