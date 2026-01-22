import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_, assert_warns
import pytest
from pytest import raises as assert_raises
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import (
from scipy.spatial.distance import pdist
from scipy.cluster._hierarchy import Heap
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close
from . import hierarchy_test_data
class TestCorrespond:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_correspond_empty(self, xp):
        y = xp.zeros((0,), dtype=xp.float64)
        Z = xp.zeros((0, 4), dtype=xp.float64)
        assert_raises(ValueError, correspond, Z, y)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_correspond_2_and_up(self, xp):
        for i in range(2, 4):
            y = np.random.rand(i * (i - 1) // 2)
            y = xp.asarray(y)
            Z = linkage(y)
            assert_(correspond(Z, y))
        for i in range(4, 15, 3):
            y = np.random.rand(i * (i - 1) // 2)
            y = xp.asarray(y)
            Z = linkage(y)
            assert_(correspond(Z, y))

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_correspond_4_and_up(self, xp):
        for i, j in list(zip(list(range(2, 4)), list(range(3, 5)))) + list(zip(list(range(3, 5)), list(range(2, 4)))):
            y = np.random.rand(i * (i - 1) // 2)
            y2 = np.random.rand(j * (j - 1) // 2)
            y = xp.asarray(y)
            y2 = xp.asarray(y2)
            Z = linkage(y)
            Z2 = linkage(y2)
            assert not correspond(Z, y2)
            assert not correspond(Z2, y)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_correspond_4_and_up_2(self, xp):
        for i, j in list(zip(list(range(2, 7)), list(range(16, 21)))) + list(zip(list(range(2, 7)), list(range(16, 21)))):
            y = np.random.rand(i * (i - 1) // 2)
            y2 = np.random.rand(j * (j - 1) // 2)
            y = xp.asarray(y)
            y2 = xp.asarray(y2)
            Z = linkage(y)
            Z2 = linkage(y2)
            assert not correspond(Z, y2)
            assert not correspond(Z2, y)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_num_obs_linkage_multi_matrix(self, xp):
        for n in range(2, 10):
            X = np.random.rand(n, 4)
            Y = pdist(X)
            Y = xp.asarray(Y)
            Z = linkage(Y)
            assert_equal(num_obs_linkage(Z), n)