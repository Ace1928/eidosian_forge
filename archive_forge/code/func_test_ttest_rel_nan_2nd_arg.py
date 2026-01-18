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
def test_ttest_rel_nan_2nd_arg():
    x = [np.nan, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 1.0, 2.0]
    r1 = stats.ttest_rel(x, y, nan_policy='omit')
    r2 = stats.ttest_rel(y, x, nan_policy='omit')
    assert_allclose(r2.statistic, -r1.statistic, atol=1e-15)
    assert_allclose(r2.pvalue, r1.pvalue, atol=1e-15)
    r3 = stats.ttest_rel(y[1:], x[1:])
    assert_allclose(r2, r3, atol=1e-15)
    assert_allclose(r2, (-2, 0.1835), atol=0.0001)