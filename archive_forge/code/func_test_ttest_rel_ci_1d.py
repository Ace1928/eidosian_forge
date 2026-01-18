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
@pytest.mark.parametrize('alternative', ['two-sided', 'less', 'greater'])
def test_ttest_rel_ci_1d(alternative):
    rng = np.random.default_rng(3749065329432213059)
    n = 10
    x = rng.normal(size=n, loc=1.5, scale=2)
    y = rng.normal(size=n, loc=2, scale=2)
    ref = {'two-sided': [-1.912194489914035, 0.400169725914035], 'greater': [-1.563944820311475, np.inf], 'less': [-np.inf, 0.05192005631147523]}
    res = stats.ttest_rel(x, y, alternative=alternative)
    ci = res.confidence_interval(confidence_level=0.85)
    assert_allclose(ci, ref[alternative])
    assert_equal(res.df, n - 1)