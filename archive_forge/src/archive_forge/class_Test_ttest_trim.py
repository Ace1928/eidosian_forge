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
class Test_ttest_trim:
    params = [[[1, 2, 3], [1.1, 2.9, 4.2], 0.5361949075312673, -0.6864951273557258, 0.2], [[56, 128.6, 12, 123.8, 64.34, 78, 763.3], [1.1, 2.9, 4.2], 0.00998909252078421, 4.591598691181999, 0.2], [[56, 128.6, 12, 123.8, 64.34, 78, 763.3], [1.1, 2.9, 4.2], 0.10512380092302633, 2.832256715395378, 0.32], [[2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9], [6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1], 0.002878909511344, -4.2461168970325, 0.2], [[-0.84504783, 0.13366078, 3.53601757, -0.62908581, 0.54119466, -1.16511574, -0.08836614, 1.18495416, 2.48028757, -1.58925028, -1.6706357, 0.3090472, -2.12258305, 0.3697304, -1.0415207, -0.57783497, -0.90997008, 1.09850192, 0.41270579, -1.4927376], [1.2725522, 1.1657899, 2.7509041, 1.2389013, -0.9490494, -1.0752459, 1.1038576, 2.9912821, 3.5349111, 0.4171922, 1.0168959, -0.7625041, -0.4300008, 3.0431921, 1.6035947, 0.5285634, -0.7649405, 1.5575896, 1.3670797, 1.1726023], 0.005293305834235, -3.0983317739483, 0.2]]

    @pytest.mark.parametrize('a,b,pr,tr,trim', params)
    def test_ttest_compare_r(self, a, b, pr, tr, trim):
        """
        Using PairedData's yuen.t.test method. Something to note is that there
        are at least 3 R packages that come with a trimmed t-test method, and
        comparisons were made between them. It was found that PairedData's
        method's results match this method, SAS, and one of the other R
        methods. A notable discrepancy was the DescTools implementation of the
        function, which only sometimes agreed with SAS, WRS2, PairedData and
        this implementation. For this reason, most comparisons in R are made
        against PairedData's method.

        Rather than providing the input and output for all evaluations, here is
        a representative example:
        > library(PairedData)
        > a <- c(1, 2, 3)
        > b <- c(1.1, 2.9, 4.2)
        > options(digits=16)
        > yuen.t.test(a, b, tr=.2)

            Two-sample Yuen test, trim=0.2

        data:  x and y
        t = -0.68649512735573, df = 3.4104431643464, p-value = 0.5361949075313
        alternative hypothesis: true difference in trimmed means is not equal
        to 0
        95 percent confidence interval:
         -3.912777195645217  2.446110528978550
        sample estimates:
        trimmed mean of x trimmed mean of y
        2.000000000000000 2.73333333333333
        """
        statistic, pvalue = stats.ttest_ind(a, b, trim=trim, equal_var=False)
        assert_allclose(statistic, tr, atol=1e-15)
        assert_allclose(pvalue, pr, atol=1e-15)

    def test_compare_SAS(self):
        a = [12, 14, 18, 25, 32, 44, 12, 14, 18, 25, 32, 44]
        b = [17, 22, 14, 12, 30, 29, 19, 17, 22, 14, 12, 30, 29, 19]
        statistic, pvalue = stats.ttest_ind(a, b, trim=0.09, equal_var=False)
        assert_allclose(pvalue, 0.514522, atol=1e-06)
        assert_allclose(statistic, 0.669169, atol=1e-06)

    def test_equal_var(self):
        """
        The PairedData library only supports unequal variances. To compare
        samples with equal variances, the multicon library is used.
        > library(multicon)
        > a <- c(2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9)
        > b <- c(6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1)
        > dv = c(a,b)
        > iv = c(rep('a', length(a)), rep('b', length(b)))
        > yuenContrast(dv~ iv, EQVAR = TRUE)
        $Ms
           N                 M wgt
        a 11 2.442857142857143   1
        b 11 5.385714285714286  -1

        $test
                              stat df              crit                   p
        results -4.246116897032513 12 2.178812829667228 0.00113508833897713
        """
        a = [2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9]
        b = [6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1]
        statistic, pvalue = stats.ttest_ind(a, b, trim=0.2)
        assert_allclose(pvalue, 0.00113508833897713, atol=1e-10)
        assert_allclose(statistic, -4.246116897032513, atol=1e-10)

    @pytest.mark.parametrize('alt,pr,tr', (('greater', 0.9985605452443, -4.2461168970325), ('less', 0.001439454755672, -4.2461168970325)))
    def test_alternatives(self, alt, pr, tr):
        """
        > library(PairedData)
        > a <- c(2.7,2.7,1.1,3.0,1.9,3.0,3.8,3.8,0.3,1.9,1.9)
        > b <- c(6.5,5.4,8.1,3.5,0.5,3.8,6.8,4.9,9.5,6.2,4.1)
        > options(digits=16)
        > yuen.t.test(a, b, alternative = 'greater')
        """
        a = [2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9]
        b = [6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1]
        statistic, pvalue = stats.ttest_ind(a, b, trim=0.2, equal_var=False, alternative=alt)
        assert_allclose(pvalue, pr, atol=1e-10)
        assert_allclose(statistic, tr, atol=1e-10)

    def test_errors_unsupported(self):
        match = 'Permutations are currently not supported with trimming.'
        with assert_raises(ValueError, match=match):
            stats.ttest_ind([1, 2], [2, 3], trim=0.2, permutations=2)

    @pytest.mark.parametrize('trim', [-0.2, 0.5, 1])
    def test_trim_bounds_error(self, trim):
        match = 'Trimming percentage should be 0 <= `trim` < .5.'
        with assert_raises(ValueError, match=match):
            stats.ttest_ind([1, 2], [2, 1], trim=trim)