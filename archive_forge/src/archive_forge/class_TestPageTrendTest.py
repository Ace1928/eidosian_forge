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
class TestPageTrendTest:
    np.random.seed(0)
    data_3_25 = np.random.rand(3, 25)
    data_10_26 = np.random.rand(10, 26)
    ts = [(12805, 0.3886487053947608, False, 'asymptotic', data_3_25), (49140, 0.02888978556179862, False, 'asymptotic', data_10_26), (12332, 0.7722477197436702, False, 'asymptotic', [[72, 47, 73, 35, 47, 96, 30, 59, 41, 36, 56, 49, 81, 43, 70, 47, 28, 28, 62, 20, 61, 20, 80, 24, 50], [68, 52, 60, 34, 44, 20, 65, 88, 21, 81, 48, 31, 31, 67, 69, 94, 30, 24, 40, 87, 70, 43, 50, 96, 43], [81, 13, 85, 35, 79, 12, 92, 86, 21, 64, 16, 64, 68, 17, 16, 89, 71, 43, 43, 36, 54, 13, 66, 51, 55]]), (266, 4.121656378600823e-05, False, 'exact', [[1.5, 4.0, 8.3, 5, 19, 11], [5, 4, 3.5, 10, 20, 21], [8.4, 3.2, 10, 12, 14, 15]]), (332, 0.9566400920502488, True, 'exact', [[4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1], [3, 4, 1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]), (241, 0.9622210164861476, True, 'exact', [[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [2, 1, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]), (197, 0.9619432897162209, True, 'exact', [[6, 5, 4, 3, 2, 1], [6, 5, 4, 3, 2, 1], [1, 3, 4, 5, 2, 6]]), (423, 0.9590458306880073, True, 'exact', [[5, 4, 3, 2, 1], [5, 4, 3, 2, 1], [5, 4, 3, 2, 1], [5, 4, 3, 2, 1], [5, 4, 3, 2, 1], [5, 4, 3, 2, 1], [4, 1, 3, 2, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]), (217, 0.9693058575034678, True, 'exact', [[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [2, 1, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]), (395, 0.991530289351305, True, 'exact', [[7, 6, 5, 4, 3, 2, 1], [7, 6, 5, 4, 3, 2, 1], [6, 5, 7, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6, 7]]), (117, 0.9997817843373017, True, 'exact', [[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [2, 1, 3], [1, 2, 3]])]

    @pytest.mark.parametrize('L, p, ranked, method, data', ts)
    def test_accuracy(self, L, p, ranked, method, data):
        np.random.seed(42)
        res = stats.page_trend_test(data, ranked=ranked, method=method)
        assert_equal(L, res.statistic)
        assert_allclose(p, res.pvalue)
        assert_equal(method, res.method)
    ts2 = [(542, 0.9481266260876332, True, 'exact', [[10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [1, 8, 4, 7, 6, 5, 9, 3, 2, 10]]), (1322, 0.9993113928199309, True, 'exact', [[10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [9, 2, 8, 7, 6, 5, 4, 3, 10, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), (2286, 0.9908688345484833, True, 'exact', [[8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1], [1, 3, 5, 6, 4, 7, 2, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]])]

    @pytest.mark.parametrize('L, p, ranked, method, data', ts)
    @pytest.mark.slow()
    def test_accuracy2(self, L, p, ranked, method, data):
        np.random.seed(42)
        res = stats.page_trend_test(data, ranked=ranked, method=method)
        assert_equal(L, res.statistic)
        assert_allclose(p, res.pvalue)
        assert_equal(method, res.method)

    def test_options(self):
        np.random.seed(42)
        m, n = (10, 20)
        predicted_ranks = np.arange(1, n + 1)
        perm = np.random.permutation(np.arange(n))
        data = np.random.rand(m, n)
        ranks = stats.rankdata(data, axis=1)
        res1 = stats.page_trend_test(ranks)
        res2 = stats.page_trend_test(ranks, ranked=True)
        res3 = stats.page_trend_test(data, ranked=False)
        res4 = stats.page_trend_test(ranks, predicted_ranks=predicted_ranks)
        res5 = stats.page_trend_test(ranks[:, perm], predicted_ranks=predicted_ranks[perm])
        assert_equal(res1.statistic, res2.statistic)
        assert_equal(res1.statistic, res3.statistic)
        assert_equal(res1.statistic, res4.statistic)
        assert_equal(res1.statistic, res5.statistic)

    def test_Ames_assay(self):
        np.random.seed(42)
        data = [[101, 117, 111], [91, 90, 107], [103, 133, 121], [136, 140, 144], [190, 161, 201], [146, 120, 116]]
        data = np.array(data).T
        predicted_ranks = np.arange(1, 7)
        res = stats.page_trend_test(data, ranked=False, predicted_ranks=predicted_ranks, method='asymptotic')
        assert_equal(res.statistic, 257)
        assert_almost_equal(res.pvalue, 0.0035, decimal=4)
        res = stats.page_trend_test(data, ranked=False, predicted_ranks=predicted_ranks, method='exact')
        assert_equal(res.statistic, 257)
        assert_almost_equal(res.pvalue, 0.0023, decimal=4)

    def test_input_validation(self):
        with assert_raises(ValueError, match='`data` must be a 2d array.'):
            stats.page_trend_test(None)
        with assert_raises(ValueError, match='`data` must be a 2d array.'):
            stats.page_trend_test([])
        with assert_raises(ValueError, match='`data` must be a 2d array.'):
            stats.page_trend_test([1, 2])
        with assert_raises(ValueError, match='`data` must be a 2d array.'):
            stats.page_trend_test([[[1]]])
        with assert_raises(ValueError, match="Page's L is only appropriate"):
            stats.page_trend_test(np.random.rand(1, 3))
        with assert_raises(ValueError, match="Page's L is only appropriate"):
            stats.page_trend_test(np.random.rand(2, 2))
        message = '`predicted_ranks` must include each integer'
        with assert_raises(ValueError, match=message):
            stats.page_trend_test(data=[[1, 2, 3], [1, 2, 3]], predicted_ranks=[0, 1, 2])
        with assert_raises(ValueError, match=message):
            stats.page_trend_test(data=[[1, 2, 3], [1, 2, 3]], predicted_ranks=[1.1, 2, 3])
        with assert_raises(ValueError, match=message):
            stats.page_trend_test(data=[[1, 2, 3], [1, 2, 3]], predicted_ranks=[1, 2, 3, 3])
        with assert_raises(ValueError, match=message):
            stats.page_trend_test(data=[[1, 2, 3], [1, 2, 3]], predicted_ranks='invalid')
        with assert_raises(ValueError, match='`data` is not properly ranked'):
            stats.page_trend_test([[0, 2, 3], [1, 2, 3]], True)
        with assert_raises(ValueError, match='`data` is not properly ranked'):
            stats.page_trend_test([[1, 2, 3], [1, 2, 4]], True)
        with assert_raises(ValueError, match='`data` contains NaNs'):
            stats.page_trend_test([[1, 2, 3], [1, 2, np.nan]], ranked=False)
        with assert_raises(ValueError, match='`method` must be in'):
            stats.page_trend_test(data=[[1, 2, 3], [1, 2, 3]], method='ekki')
        with assert_raises(TypeError, match='`ranked` must be boolean.'):
            stats.page_trend_test(data=[[1, 2, 3], [1, 2, 3]], ranked='ekki')