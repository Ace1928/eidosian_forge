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
def test_weightedtau():
    x = [12, 2, 1, 12, 2]
    y = [1, 4, 7, 1, 0]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.5669496815368272)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(x, y, additive=False)
    assert_approx_equal(tau, -0.6220571695180104)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(x, y, weigher=lambda x: 1)
    assert_approx_equal(tau, -0.47140452079103173)
    assert_equal(np.nan, p_value)
    res = stats.weightedtau(x, y)
    attributes = ('correlation', 'pvalue')
    check_named_results(res, attributes)
    assert_equal(res.correlation, res.statistic)
    tau, p_value = stats.weightedtau(x, y, rank=None)
    assert_approx_equal(tau, -0.4157652301037516)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(y, x, rank=None)
    assert_approx_equal(tau, -0.7181341329699029)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(x, y, rank=None, additive=False)
    assert_approx_equal(tau, -0.4064485096624689)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(y, x, rank=None, additive=False)
    assert_approx_equal(tau, -0.8376658293735517)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(x, y, rank=False)
    assert_approx_equal(tau, -0.5160439794026185)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(x, y, rank=True, weigher=lambda x: 1)
    assert_approx_equal(tau, -0.47140452079103173)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(y, x, rank=True, weigher=lambda x: 1)
    assert_approx_equal(tau, -0.47140452079103173)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(np.asarray(x, dtype=np.float64), y)
    assert_approx_equal(tau, -0.5669496815368272)
    tau, p_value = stats.weightedtau(np.asarray(x, dtype=np.int16), y)
    assert_approx_equal(tau, -0.5669496815368272)
    tau, p_value = stats.weightedtau(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    assert_approx_equal(tau, -0.5669496815368272)
    tau, p_value = stats.weightedtau([], [])
    assert_equal(np.nan, tau)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau([0], [0])
    assert_equal(np.nan, tau)
    assert_equal(np.nan, p_value)
    assert_raises(ValueError, stats.weightedtau, [0, 1], [0, 1, 2])
    assert_raises(ValueError, stats.weightedtau, [0, 1], [0, 1], [0])
    x = [12, 2, 1, 12, 2]
    y = [1, 4, 7, 1, np.nan]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.5669496815368272)
    x = [12, 2, np.nan, 12, 2]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.5669496815368272)
    x = [12.0, 2.0, 1.0, 12.0, 2.0]
    y = [1.0, 4.0, 7.0, 1.0, np.nan]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.5669496815368272)
    x = [12.0, 2.0, np.nan, 12.0, 2.0]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.5669496815368272)
    x = [12.0, 2.0, 1.0, 12.0, 1.0]
    y = [1.0, 4.0, 7.0, 1.0, 1.0]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.6615242347139803)
    x = [12.0, 2.0, np.nan, 12.0, np.nan]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.6615242347139803)
    y = [np.nan, 4.0, 7.0, np.nan, np.nan]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.6615242347139803)