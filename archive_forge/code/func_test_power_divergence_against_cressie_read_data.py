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
def test_power_divergence_against_cressie_read_data():
    obs = np.array([15, 11, 14, 17, 5, 11, 10, 4, 8, 10, 7, 9, 11, 3, 6, 1, 1, 4])
    beta = -0.083769
    i = np.arange(1, len(obs) + 1)
    alpha = np.log(obs.sum() / np.exp(beta * i).sum())
    expected_counts = np.exp(alpha + beta * i)
    table4 = np.vstack((obs, expected_counts)).T
    table5 = np.array([-10.0, 72200.0, -5.0, 289.0, -3.0, 65.6, -2.0, 40.6, -1.5, 34.0, -1.0, 29.5, -0.5, 26.5, 0.0, 24.6, 0.5, 23.4, 0.67, 23.1, 1.0, 22.7, 1.5, 22.6, 2.0, 22.9, 3.0, 24.8, 5.0, 35.5, 10.0, 214.0]).reshape(-1, 2)
    for lambda_, expected_stat in table5:
        stat, p = stats.power_divergence(table4[:, 0], table4[:, 1], lambda_=lambda_)
        assert_allclose(stat, expected_stat, rtol=0.005)