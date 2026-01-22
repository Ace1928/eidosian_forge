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
class TestCdfDistanceValidation:
    """
    Test that _cdf_distance() (via wasserstein_distance()) raises ValueErrors
    for bad inputs.
    """

    def test_distinct_value_and_weight_lengths(self):
        assert_raises(ValueError, stats.wasserstein_distance, [1], [2], [4], [3, 1])
        assert_raises(ValueError, stats.wasserstein_distance, [1], [2], [1, 0])

    def test_zero_weight(self):
        assert_raises(ValueError, stats.wasserstein_distance, [0, 1], [2], [0, 0])
        assert_raises(ValueError, stats.wasserstein_distance, [0, 1], [2], [3, 1], [0])

    def test_negative_weights(self):
        assert_raises(ValueError, stats.wasserstein_distance, [0, 1], [2, 2], [1, 1], [3, -1])

    def test_empty_distribution(self):
        assert_raises(ValueError, stats.wasserstein_distance, [], [2, 2])
        assert_raises(ValueError, stats.wasserstein_distance, [1], [])

    def test_inf_weight(self):
        assert_raises(ValueError, stats.wasserstein_distance, [1, 2, 1], [1, 1], [1, np.inf, 1], [1, 1])