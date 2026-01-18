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
@pytest.mark.xslow
@pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
@pytest.mark.parametrize('method', ('permutation', 'monte_carlo'))
def test_resampling_pvalue(self, method, alternative):
    rng = np.random.default_rng(24623935790378923)
    size = 100 if method == 'permutation' else 1000
    x = rng.normal(size=size)
    y = rng.normal(size=size)
    methods = {'permutation': stats.PermutationMethod(random_state=rng), 'monte_carlo': stats.MonteCarloMethod(rvs=(rng.normal,) * 2)}
    method = methods[method]
    res = stats.pearsonr(x, y, alternative=alternative, method=method)
    ref = stats.pearsonr(x, y, alternative=alternative)
    assert_allclose(res.statistic, ref.statistic, rtol=1e-15)
    assert_allclose(res.pvalue, ref.pvalue, rtol=0.01, atol=0.001)