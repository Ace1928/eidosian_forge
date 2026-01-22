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
class TestMood:

    @pytest.mark.parametrize('x,y,alternative,stat_expect,p_expect', mood_cases_with_ties())
    def test_against_SAS(self, x, y, alternative, stat_expect, p_expect):
        """
        Example code used to generate SAS output:
        DATA myData;
        INPUT X Y;
        CARDS;
        1 0
        1 1
        1 2
        1 3
        1 4
        2 0
        2 1
        2 4
        2 9
        2 16
        ods graphics on;
        proc npar1way mood data=myData ;
           class X;
            ods output  MoodTest=mt;
        proc contents data=mt;
        proc print data=mt;
          format     Prob1 17.16 Prob2 17.16 Statistic 17.16 Z 17.16 ;
            title "Mood Two-Sample Test";
        proc print data=myData;
            title "Data for above results";
          run;
        """
        statistic, pvalue = stats.mood(x, y, alternative=alternative)
        assert_allclose(stat_expect, statistic, atol=1e-16)
        assert_allclose(p_expect, pvalue, atol=1e-16)

    @pytest.mark.parametrize('alternative, expected', [('two-sided', (1.01993853354993, 0.307757612977876)), ('less', (1.01993853354993, 1 - 0.153878806488938)), ('greater', (1.01993853354993, 0.153878806488938))])
    def test_against_SAS_2(self, alternative, expected):
        x = [111, 107, 100, 99, 102, 106, 109, 108, 104, 99, 101, 96, 97, 102, 107, 113, 116, 113, 110, 98]
        y = [107, 108, 106, 98, 105, 103, 110, 105, 104, 100, 96, 108, 103, 104, 114, 114, 113, 108, 106, 99]
        res = stats.mood(x, y, alternative=alternative)
        assert_allclose(res, expected)

    def test_mood_order_of_args(self):
        np.random.seed(1234)
        x1 = np.random.randn(10, 1)
        x2 = np.random.randn(15, 1)
        z1, p1 = stats.mood(x1, x2)
        z2, p2 = stats.mood(x2, x1)
        assert_array_almost_equal([z1, p1], [-z2, p2])

    def test_mood_with_axis_none(self):
        x1 = [-0.626453810742332, 0.183643324222082, -0.835628612410047, 1.59528080213779, 0.329507771815361, -0.820468384118015, 0.487429052428485, 0.738324705129217, 0.575781351653492, -0.305388387156356, 1.51178116845085, 0.389843236411431, -0.621240580541804, -2.2146998871775, 1.12493091814311, -0.0449336090152309, -0.0161902630989461, 0.943836210685299, 0.821221195098089, 0.593901321217509]
        x2 = [-0.896914546624981, 0.184849184646742, 1.58784533120882, -1.13037567424629, -0.0802517565509893, 0.132420284381094, 0.707954729271733, -0.23969802417184, 1.98447393665293, -0.138787012119665, 0.417650750792556, 0.981752777463662, -0.392695355503813, -1.03966897694891, 1.78222896030858, -2.31106908460517, 0.878604580921265, 0.035806718015226, 1.01282869212708, 0.432265154539617, 2.09081920524915, -1.19992581964387, 1.58963820029007, 1.95465164222325, 0.00493777682814261, -2.45170638784613, 0.477237302613617, -0.596558168631403, 0.792203270299649, 0.289636710177348]
        x1 = np.array(x1)
        x2 = np.array(x2)
        x1.shape = (10, 2)
        x2.shape = (15, 2)
        assert_array_almost_equal(stats.mood(x1, x2, axis=None), [-1.31716607555, 0.18778296257])

    def test_mood_2d(self):
        ny = 5
        np.random.seed(1234)
        x1 = np.random.randn(10, ny)
        x2 = np.random.randn(15, ny)
        z_vectest, pval_vectest = stats.mood(x1, x2)
        for j in range(ny):
            assert_array_almost_equal([z_vectest[j], pval_vectest[j]], stats.mood(x1[:, j], x2[:, j]))
        x1 = x1.transpose()
        x2 = x2.transpose()
        z_vectest, pval_vectest = stats.mood(x1, x2, axis=1)
        for i in range(ny):
            assert_array_almost_equal([z_vectest[i], pval_vectest[i]], stats.mood(x1[i, :], x2[i, :]))

    def test_mood_3d(self):
        shape = (10, 5, 6)
        np.random.seed(1234)
        x1 = np.random.randn(*shape)
        x2 = np.random.randn(*shape)
        for axis in range(3):
            z_vectest, pval_vectest = stats.mood(x1, x2, axis=axis)
            axes_idx = ([1, 2], [0, 2], [0, 1])
            for i in range(shape[axes_idx[axis][0]]):
                for j in range(shape[axes_idx[axis][1]]):
                    if axis == 0:
                        slice1 = x1[:, i, j]
                        slice2 = x2[:, i, j]
                    elif axis == 1:
                        slice1 = x1[i, :, j]
                        slice2 = x2[i, :, j]
                    else:
                        slice1 = x1[i, j, :]
                        slice2 = x2[i, j, :]
                    assert_array_almost_equal([z_vectest[i, j], pval_vectest[i, j]], stats.mood(slice1, slice2))

    def test_mood_bad_arg(self):
        assert_raises(ValueError, stats.mood, [1], [])

    def test_mood_alternative(self):
        np.random.seed(0)
        x = stats.norm.rvs(scale=0.75, size=100)
        y = stats.norm.rvs(scale=1.25, size=100)
        stat1, p1 = stats.mood(x, y, alternative='two-sided')
        stat2, p2 = stats.mood(x, y, alternative='less')
        stat3, p3 = stats.mood(x, y, alternative='greater')
        assert stat1 == stat2 == stat3
        assert_allclose(p1, 0, atol=1e-07)
        assert_allclose(p2, p1 / 2)
        assert_allclose(p3, 1 - p1 / 2)
        with pytest.raises(ValueError, match='alternative must be...'):
            stats.mood(x, y, alternative='ekki-ekki')

    @pytest.mark.parametrize('alternative', ['two-sided', 'less', 'greater'])
    def test_result(self, alternative):
        rng = np.random.default_rng(265827767938813079281100964083953437622)
        x1 = rng.standard_normal((10, 1))
        x2 = rng.standard_normal((15, 1))
        res = stats.mood(x1, x2, alternative=alternative)
        assert_equal((res.statistic, res.pvalue), res)