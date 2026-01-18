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
def test_tvar(self):
    y = stats.tvar(X, limits=(2, 8), inclusive=(True, True))
    assert_approx_equal(y, 4.666666666666666, significant=self.dprec)
    y = stats.tvar(X, limits=None)
    assert_approx_equal(y, X.var(ddof=1), significant=self.dprec)
    x_2d = arange(63, dtype=float64).reshape((9, 7))
    y = stats.tvar(x_2d, axis=None)
    assert_approx_equal(y, x_2d.var(ddof=1), significant=self.dprec)
    y = stats.tvar(x_2d, axis=0)
    assert_array_almost_equal(y[0], np.full((1, 7), 367.5), decimal=8)
    y = stats.tvar(x_2d, axis=1)
    assert_array_almost_equal(y[0], np.full((1, 9), 4.66666667), decimal=8)
    y = stats.tvar(x_2d[3, :])
    assert_approx_equal(y, 4.666666666666667, significant=self.dprec)
    with suppress_warnings() as sup:
        sup.record(RuntimeWarning, 'Degrees of freedom <= 0 for slice.')
        y = stats.tvar(x_2d, limits=(1, 5), axis=1, inclusive=(True, True))
        assert_approx_equal(y[0], 2.5, significant=self.dprec)
        y = stats.tvar(x_2d, limits=(0, 6), axis=1, inclusive=(True, True))
        assert_approx_equal(y[0], 4.666666666666667, significant=self.dprec)
        assert_equal(y[1], np.nan)