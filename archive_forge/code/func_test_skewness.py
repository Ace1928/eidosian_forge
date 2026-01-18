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
def test_skewness(self):
    y = stats.skew(self.scalar_testcase)
    assert np.isnan(y)
    y = stats.skew(self.testmathworks)
    assert_approx_equal(y, -0.29322304336607, 10)
    y = stats.skew(self.testmathworks, bias=0)
    assert_approx_equal(y, -0.43711110502394, 10)
    y = stats.skew(self.testcase)
    assert_approx_equal(y, 0.0, 10)
    x = np.arange(10.0)
    x[9] = np.nan
    with np.errstate(invalid='ignore'):
        assert_equal(stats.skew(x), np.nan)
    assert_equal(stats.skew(x, nan_policy='omit'), 0.0)
    assert_raises(ValueError, stats.skew, x, nan_policy='raise')
    assert_raises(ValueError, stats.skew, x, nan_policy='foobar')