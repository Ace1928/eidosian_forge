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
class TestRatioUniforms:
    """ Tests for rvs_ratio_uniforms are in test_sampling.py,
    as rvs_ratio_uniforms is deprecated and moved to stats.sampling """

    def test_consistency(self):
        f = stats.norm.pdf
        v = np.sqrt(f(np.sqrt(2))) * np.sqrt(2)
        umax = np.sqrt(f(0))
        gen = stats.sampling.RatioUniforms(f, umax=umax, vmin=-v, vmax=v, random_state=12345)
        r1 = gen.rvs(10)
        deprecation_msg = 'Please use `RatioUniforms` from the `scipy.stats.sampling` namespace.'
        with pytest.warns(DeprecationWarning, match=deprecation_msg):
            r2 = stats.rvs_ratio_uniforms(f, umax, -v, v, size=10, random_state=12345)
        assert_equal(r1, r2)