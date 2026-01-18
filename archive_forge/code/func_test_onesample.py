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
def test_onesample(self):
    with suppress_warnings() as sup, np.errstate(invalid='ignore', divide='ignore'):
        sup.filter(RuntimeWarning, 'Degrees of freedom <= 0 for slice')
        t, p = stats.ttest_1samp(4.0, 3.0)
    assert_(np.isnan(t))
    assert_(np.isnan(p))
    t, p = stats.ttest_1samp(self.X1, 0)
    assert_array_almost_equal(t, self.T1_0)
    assert_array_almost_equal(p, self.P1_0)
    res = stats.ttest_1samp(self.X1, 0)
    attributes = ('statistic', 'pvalue')
    check_named_results(res, attributes)
    t, p = stats.ttest_1samp(self.X2, 0)
    assert_array_almost_equal(t, self.T2_0)
    assert_array_almost_equal(p, self.P2_0)
    t, p = stats.ttest_1samp(self.X1, 1)
    assert_array_almost_equal(t, self.T1_1)
    assert_array_almost_equal(p, self.P1_1)
    t, p = stats.ttest_1samp(self.X1, 2)
    assert_array_almost_equal(t, self.T1_2)
    assert_array_almost_equal(p, self.P1_2)
    x = stats.norm.rvs(loc=5, scale=10, size=51, random_state=7654567)
    x[50] = np.nan
    with np.errstate(invalid='ignore'):
        assert_array_equal(stats.ttest_1samp(x, 5.0), (np.nan, np.nan))
        assert_array_almost_equal(stats.ttest_1samp(x, 5.0, nan_policy='omit'), (-1.641262407436716, 0.107147027334048))
        assert_raises(ValueError, stats.ttest_1samp, x, 5.0, nan_policy='raise')
        assert_raises(ValueError, stats.ttest_1samp, x, 5.0, nan_policy='foobar')