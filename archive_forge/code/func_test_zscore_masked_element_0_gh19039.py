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
def test_zscore_masked_element_0_gh19039(self):
    rng = np.random.default_rng(8675309)
    x = rng.standard_normal(10)
    mask = np.zeros_like(x)
    y = np.ma.masked_array(x, mask)
    y.mask[0] = True
    ref = stats.zscore(x[1:])
    assert not np.any(np.isnan(ref))
    res = stats.zscore(y)
    assert_allclose(res[1:], ref)
    res = stats.zscore(y, axis=None)
    assert_allclose(res[1:], ref)
    y[1:] = y[1]
    res = stats.zscore(y)
    assert_equal(res[1:], np.nan)
    res = stats.zscore(y, axis=None)
    assert_equal(res[1:], np.nan)