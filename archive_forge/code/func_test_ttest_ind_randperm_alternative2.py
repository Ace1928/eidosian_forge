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
@pytest.mark.slow()
def test_ttest_ind_randperm_alternative2(self):
    np.random.seed(0)
    N = 50
    a = np.random.rand(N, 4)
    b = np.random.rand(N, 4)
    options_p = {'permutations': 20000, 'random_state': 0}
    options_p.update(alternative='greater')
    res_g_ab = stats.ttest_ind(a, b, **options_p)
    options_p.update(alternative='less')
    res_l_ab = stats.ttest_ind(a, b, **options_p)
    options_p.update(alternative='two-sided')
    res_2_ab = stats.ttest_ind(a, b, **options_p)
    assert_equal(res_g_ab.pvalue + res_l_ab.pvalue, 1 + 1 / (options_p['permutations'] + 1))
    mask = res_g_ab.pvalue <= 0.5
    assert_allclose(2 * res_g_ab.pvalue[mask], res_2_ab.pvalue[mask], atol=0.02)
    assert_allclose(2 * (1 - res_g_ab.pvalue[~mask]), res_2_ab.pvalue[~mask], atol=0.02)
    assert_allclose(2 * res_l_ab.pvalue[~mask], res_2_ab.pvalue[~mask], atol=0.02)
    assert_allclose(2 * (1 - res_l_ab.pvalue[mask]), res_2_ab.pvalue[mask], atol=0.02)