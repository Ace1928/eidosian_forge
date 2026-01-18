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
def test_zscore_constant_input_2d(self):
    x = np.array([[10.0, 10.0, 10.0, 10.0], [10.0, 11.0, 12.0, 13.0]])
    z0 = stats.zscore(x, axis=0)
    assert_equal(z0, np.array([[np.nan, -1.0, -1.0, -1.0], [np.nan, 1.0, 1.0, 1.0]]))
    z1 = stats.zscore(x, axis=1)
    assert_equal(z1, np.array([[np.nan, np.nan, np.nan, np.nan], stats.zscore(x[1])]))
    z = stats.zscore(x, axis=None)
    assert_equal(z, stats.zscore(x.ravel()).reshape(x.shape))
    y = np.ones((3, 6))
    z = stats.zscore(y, axis=None)
    assert_equal(z, np.full(y.shape, np.nan))