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
class TestMGCErrorWarnings:
    """ Tests errors and warnings derived from MGC.
    """

    def test_error_notndarray(self):
        x = np.arange(20)
        y = [5] * 20
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)
        assert_raises(ValueError, stats.multiscale_graphcorr, y, x)

    def test_error_shape(self):
        x = np.arange(100).reshape(25, 4)
        y = x.reshape(10, 10)
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)

    def test_error_lowsamples(self):
        x = np.arange(3)
        y = np.arange(3)
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)

    def test_error_nans(self):
        x = np.arange(20, dtype=float)
        x[0] = np.nan
        assert_raises(ValueError, stats.multiscale_graphcorr, x, x)
        y = np.arange(20)
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)

    def test_error_wrongdisttype(self):
        x = np.arange(20)
        compute_distance = 0
        assert_raises(ValueError, stats.multiscale_graphcorr, x, x, compute_distance=compute_distance)

    @pytest.mark.parametrize('reps', [-1, '1'])
    def test_error_reps(self, reps):
        x = np.arange(20)
        assert_raises(ValueError, stats.multiscale_graphcorr, x, x, reps=reps)

    def test_warns_reps(self):
        x = np.arange(20)
        reps = 100
        assert_warns(RuntimeWarning, stats.multiscale_graphcorr, x, x, reps=reps)

    def test_error_infty(self):
        x = np.arange(20)
        y = np.ones(20) * np.inf
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)