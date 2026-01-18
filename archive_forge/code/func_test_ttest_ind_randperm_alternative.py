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
def test_ttest_ind_randperm_alternative(self):
    np.random.seed(0)
    N = 50
    a = np.random.rand(2, 3, N)
    b = np.random.rand(3, N)
    options_p = {'axis': -1, 'permutations': 1000, 'random_state': 0}
    options_p.update(alternative='greater')
    res_g_ab = stats.ttest_ind(a, b, **options_p)
    res_g_ba = stats.ttest_ind(b, a, **options_p)
    options_p.update(alternative='less')
    res_l_ab = stats.ttest_ind(a, b, **options_p)
    res_l_ba = stats.ttest_ind(b, a, **options_p)
    assert_equal(res_g_ab.statistic, res_l_ab.statistic)
    assert_equal(res_g_ab.statistic, -res_g_ba.statistic)
    assert_equal(res_l_ab.statistic, -res_l_ba.statistic)
    assert_equal(res_g_ab.pvalue + res_l_ab.pvalue, 1 + 1 / (options_p['permutations'] + 1))
    assert_equal(res_g_ba.pvalue + res_l_ba.pvalue, 1 + 1 / (options_p['permutations'] + 1))