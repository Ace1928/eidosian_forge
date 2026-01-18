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
def test_ttest_ind_permutation_nanpolicy(self):
    np.random.seed(0)
    N = 50
    a = np.random.rand(N, 5)
    b = np.random.rand(N, 5)
    a[5, 1] = np.nan
    b[8, 2] = np.nan
    a[9, 3] = np.nan
    b[9, 3] = np.nan
    options_p = {'permutations': 1000, 'random_state': 0}
    options_p.update(nan_policy='raise')
    with assert_raises(ValueError, match='The input contains nan values'):
        res = stats.ttest_ind(a, b, **options_p)
    with suppress_warnings() as sup:
        sup.record(RuntimeWarning, 'invalid value*')
        options_p.update(nan_policy='propagate')
        res = stats.ttest_ind(a, b, **options_p)
        mask = np.isnan(a).any(axis=0) | np.isnan(b).any(axis=0)
        res2 = stats.ttest_ind(a[:, ~mask], b[:, ~mask], **options_p)
        assert_equal(res.pvalue[mask], np.nan)
        assert_equal(res.statistic[mask], np.nan)
        assert_allclose(res.pvalue[~mask], res2.pvalue)
        assert_allclose(res.statistic[~mask], res2.statistic)
        res = stats.ttest_ind(a.ravel(), b.ravel(), **options_p)
        assert np.isnan(res.pvalue)
        assert np.isnan(res.statistic)