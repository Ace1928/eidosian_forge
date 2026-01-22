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
class TestIsMonotonic:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_monotonic_empty(self, xp):
        Z = xp.zeros((0, 4), dtype=xp.float64)
        assert_raises(ValueError, is_monotonic, Z)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_monotonic_1x4(self, xp):
        Z = xp.asarray([[0, 1, 0.3, 2]], dtype=xp.float64)
        assert is_monotonic(Z)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_monotonic_2x4_T(self, xp):
        Z = xp.asarray([[0, 1, 0.3, 2], [2, 3, 0.4, 3]], dtype=xp.float64)
        assert is_monotonic(Z)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_monotonic_2x4_F(self, xp):
        Z = xp.asarray([[0, 1, 0.4, 2], [2, 3, 0.3, 3]], dtype=xp.float64)
        assert not is_monotonic(Z)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_monotonic_3x4_T(self, xp):
        Z = xp.asarray([[0, 1, 0.3, 2], [2, 3, 0.4, 2], [4, 5, 0.6, 4]], dtype=xp.float64)
        assert is_monotonic(Z)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_monotonic_3x4_F1(self, xp):
        Z = xp.asarray([[0, 1, 0.3, 2], [2, 3, 0.2, 2], [4, 5, 0.6, 4]], dtype=xp.float64)
        assert not is_monotonic(Z)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_monotonic_3x4_F2(self, xp):
        Z = xp.asarray([[0, 1, 0.8, 2], [2, 3, 0.4, 2], [4, 5, 0.6, 4]], dtype=xp.float64)
        assert not is_monotonic(Z)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_monotonic_3x4_F3(self, xp):
        Z = xp.asarray([[0, 1, 0.3, 2], [2, 3, 0.4, 2], [4, 5, 0.2, 4]], dtype=xp.float64)
        assert not is_monotonic(Z)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_monotonic_tdist_linkage1(self, xp):
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
        assert is_monotonic(Z)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_monotonic_tdist_linkage2(self, xp):
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
        Z[2, 2] = 0.0
        assert not is_monotonic(Z)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_monotonic_Q_linkage(self, xp):
        X = xp.asarray(hierarchy_test_data.Q_X)
        Z = linkage(X, 'single')
        assert is_monotonic(Z)