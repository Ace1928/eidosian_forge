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
class TestCombinePvalues:

    def test_fisher(self):
        xsq, p = stats.combine_pvalues([0.01, 0.2, 0.3], method='fisher')
        assert_approx_equal(p, 0.02156, significant=4)

    def test_stouffer(self):
        Z, p = stats.combine_pvalues([0.01, 0.2, 0.3], method='stouffer')
        assert_approx_equal(p, 0.01651, significant=4)

    def test_stouffer2(self):
        Z, p = stats.combine_pvalues([0.5, 0.5, 0.5], method='stouffer')
        assert_approx_equal(p, 0.5, significant=4)

    def test_weighted_stouffer(self):
        Z, p = stats.combine_pvalues([0.01, 0.2, 0.3], method='stouffer', weights=np.ones(3))
        assert_approx_equal(p, 0.01651, significant=4)

    def test_weighted_stouffer2(self):
        Z, p = stats.combine_pvalues([0.01, 0.2, 0.3], method='stouffer', weights=np.array((1, 4, 9)))
        assert_approx_equal(p, 0.1464, significant=4)

    def test_pearson(self):
        Z, p = stats.combine_pvalues([0.01, 0.2, 0.3], method='pearson')
        assert_approx_equal(p, 0.02213, significant=4)

    def test_tippett(self):
        Z, p = stats.combine_pvalues([0.01, 0.2, 0.3], method='tippett')
        assert_approx_equal(p, 0.0297, significant=4)

    def test_mudholkar_george(self):
        Z, p = stats.combine_pvalues([0.1, 0.1, 0.1], method='mudholkar_george')
        assert_approx_equal(p, 0.019462, significant=4)

    def test_mudholkar_george_equal_fisher_pearson_average(self):
        Z, p = stats.combine_pvalues([0.01, 0.2, 0.3], method='mudholkar_george')
        Z_f, p_f = stats.combine_pvalues([0.01, 0.2, 0.3], method='fisher')
        Z_p, p_p = stats.combine_pvalues([0.01, 0.2, 0.3], method='pearson')
        assert_approx_equal(0.5 * (Z_f + Z_p), Z, significant=4)
    methods = ['fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george']

    @pytest.mark.parametrize('variant', ['single', 'all', 'random'])
    @pytest.mark.parametrize('method', methods)
    def test_monotonicity(self, variant, method):
        m, n = (10, 7)
        rng = np.random.default_rng(278448169958891062669391462690811630763)
        if variant == 'single':
            pvaluess = np.full((m, n), rng.random(n))
            pvaluess[:, 0] = np.linspace(0.1, 0.9, m)
        elif variant == 'all':
            pvaluess = np.full((n, m), np.linspace(0.1, 0.9, m)).T
        elif variant == 'random':
            pvaluess = np.sort(rng.uniform(0, 1, size=(m, n)), axis=0)
        combined_pvalues = [stats.combine_pvalues(pvalues, method=method)[1] for pvalues in pvaluess]
        assert np.all(np.diff(combined_pvalues) >= 0)

    @pytest.mark.parametrize('method', methods)
    def test_result(self, method):
        res = stats.combine_pvalues([0.01, 0.2, 0.3], method=method)
        assert_equal((res.statistic, res.pvalue), res)