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
def test_tmin(self):
    assert_equal(stats.tmin(4), 4)
    x = np.arange(10)
    assert_equal(stats.tmin(x), 0)
    assert_equal(stats.tmin(x, lowerlimit=0), 0)
    assert_equal(stats.tmin(x, lowerlimit=0, inclusive=False), 1)
    x = x.reshape((5, 2))
    assert_equal(stats.tmin(x, lowerlimit=0, inclusive=False), [2, 1])
    assert_equal(stats.tmin(x, axis=1), [0, 2, 4, 6, 8])
    assert_equal(stats.tmin(x, axis=None), 0)
    x = np.arange(10.0)
    x[9] = np.nan
    with suppress_warnings() as sup:
        sup.record(RuntimeWarning, 'invalid value*')
        assert_equal(stats.tmin(x), np.nan)
        assert_equal(stats.tmin(x, nan_policy='omit'), 0.0)
        assert_raises(ValueError, stats.tmin, x, nan_policy='raise')
        assert_raises(ValueError, stats.tmin, x, nan_policy='foobar')
        msg = "'propagate', 'raise', 'omit'"
        with assert_raises(ValueError, match=msg):
            stats.tmin(x, nan_policy='foo')
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'All-NaN slice encountered')
        x = np.arange(16).reshape(4, 4)
        res = stats.tmin(x, lowerlimit=4, axis=1)
        assert_equal(res, [np.nan, 4, 8, 12])