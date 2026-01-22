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
class TestGeometricStandardDeviation:
    array_1d = np.arange(2 * 3 * 4) + 1
    gstd_array_1d = 2.294407613602
    array_3d = array_1d.reshape(2, 3, 4)

    def test_1d_array(self):
        gstd_actual = stats.gstd(self.array_1d)
        assert_allclose(gstd_actual, self.gstd_array_1d)

    def test_1d_numeric_array_like_input(self):
        gstd_actual = stats.gstd(tuple(self.array_1d))
        assert_allclose(gstd_actual, self.gstd_array_1d)

    def test_raises_value_error_non_array_like_input(self):
        with pytest.raises(ValueError, match='Invalid array input'):
            stats.gstd('This should fail as it can not be cast to an array.')

    def test_raises_value_error_zero_entry(self):
        with pytest.raises(ValueError, match='Non positive value'):
            stats.gstd(np.append(self.array_1d, [0]))

    def test_raises_value_error_negative_entry(self):
        with pytest.raises(ValueError, match='Non positive value'):
            stats.gstd(np.append(self.array_1d, [-1]))

    def test_raises_value_error_inf_entry(self):
        with pytest.raises(ValueError, match='Infinite value'):
            stats.gstd(np.append(self.array_1d, [np.inf]))

    def test_propagates_nan_values(self):
        a = array([[1, 1, 1, 16], [np.nan, 1, 2, 3]])
        gstd_actual = stats.gstd(a, axis=1)
        assert_allclose(gstd_actual, np.array([4, np.nan]))

    def test_ddof_equal_to_number_of_observations(self):
        with pytest.raises(ValueError, match='Degrees of freedom <= 0'):
            stats.gstd(self.array_1d, ddof=self.array_1d.size)

    def test_3d_array(self):
        gstd_actual = stats.gstd(self.array_3d, axis=None)
        assert_allclose(gstd_actual, self.gstd_array_1d)

    def test_3d_array_axis_type_tuple(self):
        gstd_actual = stats.gstd(self.array_3d, axis=(1, 2))
        assert_allclose(gstd_actual, [2.12939215, 1.22120169])

    def test_3d_array_axis_0(self):
        gstd_actual = stats.gstd(self.array_3d, axis=0)
        gstd_desired = np.array([[6.1330555493918, 3.95890021012, 3.1206598248344, 2.6651441426902], [2.3758135028411, 2.174581428192, 2.0260062829505, 1.9115518327308], [1.8205343606803, 1.746342404566, 1.6846557065742, 1.6325269194382]])
        assert_allclose(gstd_actual, gstd_desired)

    def test_3d_array_axis_1(self):
        gstd_actual = stats.gstd(self.array_3d, axis=1)
        gstd_desired = np.array([[3.118993630946, 2.275985934063, 1.933995977619, 1.742896469724], [1.271693593916, 1.254158641801, 1.238774141609, 1.225164057869]])
        assert_allclose(gstd_actual, gstd_desired)

    def test_3d_array_axis_2(self):
        gstd_actual = stats.gstd(self.array_3d, axis=2)
        gstd_desired = np.array([[1.8242475707664, 1.2243686572447, 1.1318311657788], [1.0934830582351, 1.0724479791887, 1.0591498540749]])
        assert_allclose(gstd_actual, gstd_desired)

    def test_masked_3d_array(self):
        ma = np.ma.masked_where(self.array_3d > 16, self.array_3d)
        gstd_actual = stats.gstd(ma, axis=2)
        gstd_desired = stats.gstd(self.array_3d, axis=2)
        mask = [[0, 0, 0], [0, 1, 1]]
        assert_allclose(gstd_actual, gstd_desired)
        assert_equal(gstd_actual.mask, mask)