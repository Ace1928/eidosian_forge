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
def test_theilslopes():
    slope, intercept, lower, upper = stats.theilslopes([0, 1, 1])
    assert_almost_equal(slope, 0.5)
    assert_almost_equal(intercept, 0.5)
    msg = "method must be either 'joint' or 'separate'.'joint_separate' is invalid."
    with pytest.raises(ValueError, match=msg):
        stats.theilslopes([0, 1, 1], method='joint_separate')
    slope, intercept, lower, upper = stats.theilslopes([0, 1, 1], method='joint')
    assert_almost_equal(slope, 0.5)
    assert_almost_equal(intercept, 0.0)
    x = [1, 2, 3, 4, 10, 12, 18]
    y = [9, 15, 19, 20, 45, 55, 78]
    slope, intercept, lower, upper = stats.theilslopes(y, x, 0.07, method='separate')
    assert_almost_equal(slope, 4)
    assert_almost_equal(intercept, 4.0)
    assert_almost_equal(upper, 4.38, decimal=2)
    assert_almost_equal(lower, 3.71, decimal=2)
    slope, intercept, lower, upper = stats.theilslopes(y, x, 0.07, method='joint')
    assert_almost_equal(slope, 4)
    assert_almost_equal(intercept, 6.0)
    assert_almost_equal(upper, 4.38, decimal=2)
    assert_almost_equal(lower, 3.71, decimal=2)