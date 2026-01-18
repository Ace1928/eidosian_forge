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
def test_trimboth(self):
    a = np.arange(11)
    assert_equal(np.sort(stats.trimboth(a, 3 / 11.0)), np.arange(3, 8))
    assert_equal(np.sort(stats.trimboth(a, 0.2)), np.array([2, 3, 4, 5, 6, 7, 8]))
    assert_equal(np.sort(stats.trimboth(np.arange(24).reshape(6, 4), 0.2)), np.arange(4, 20).reshape(4, 4))
    assert_equal(np.sort(stats.trimboth(np.arange(24).reshape(4, 6).T, 2 / 6.0)), np.array([[2, 8, 14, 20], [3, 9, 15, 21]]))
    assert_raises(ValueError, stats.trimboth, np.arange(24).reshape(4, 6).T, 4 / 6.0)
    assert_equal(stats.trimboth([], 0.1), [])
    assert_equal(stats.trimboth([], 3 / 11.0), [])
    assert_equal(stats.trimboth([], 4 / 6.0), [])