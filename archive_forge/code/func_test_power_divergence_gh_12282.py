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
def test_power_divergence_gh_12282(self):
    f_obs = np.array([[10, 20], [30, 20]])
    f_exp = np.array([[5, 15], [35, 25]])
    with assert_raises(ValueError, match='For each axis slice...'):
        stats.power_divergence(f_obs=[10, 20], f_exp=[30, 60])
    with assert_raises(ValueError, match='For each axis slice...'):
        stats.power_divergence(f_obs=f_obs, f_exp=f_exp, axis=1)
    stat, pval = stats.power_divergence(f_obs=f_obs, f_exp=f_exp)
    assert_allclose(stat, [5.71428571, 2.66666667])
    assert_allclose(pval, [0.01682741, 0.10247043])