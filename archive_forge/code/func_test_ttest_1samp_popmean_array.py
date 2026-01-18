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
def test_ttest_1samp_popmean_array():
    rng = np.random.default_rng(2913300596553337193)
    x = rng.random(size=(1, 15, 20))
    message = '`popmean.shape\\[axis\\]` must equal 1.'
    popmean = rng.random(size=(5, 2, 20))
    with pytest.raises(ValueError, match=message):
        stats.ttest_1samp(x, popmean=popmean, axis=-2)
    popmean = rng.random(size=(5, 1, 20))
    res = stats.ttest_1samp(x, popmean=popmean, axis=-2)
    assert res.statistic.shape == (5, 20)
    ci = np.expand_dims(res.confidence_interval(), axis=-2)
    res = stats.ttest_1samp(x, popmean=ci, axis=-2)
    assert_allclose(res.pvalue, 0.05)