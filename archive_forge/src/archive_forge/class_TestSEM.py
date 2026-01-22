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
class TestSEM:
    testcase = [1, 2, 3, 4]
    scalar_testcase = 4.0

    def test_sem(self):
        with suppress_warnings() as sup, np.errstate(invalid='ignore'):
            sup.filter(RuntimeWarning, 'Degrees of freedom <= 0 for slice')
            y = stats.sem(self.scalar_testcase)
        assert_(np.isnan(y))
        y = stats.sem(self.testcase)
        assert_approx_equal(y, 0.6454972244)
        n = len(self.testcase)
        assert_allclose(stats.sem(self.testcase, ddof=0) * np.sqrt(n / (n - 2)), stats.sem(self.testcase, ddof=2))
        x = np.arange(10.0)
        x[9] = np.nan
        assert_equal(stats.sem(x), np.nan)
        assert_equal(stats.sem(x, nan_policy='omit'), 0.9128709291752769)
        assert_raises(ValueError, stats.sem, x, nan_policy='raise')
        assert_raises(ValueError, stats.sem, x, nan_policy='foobar')