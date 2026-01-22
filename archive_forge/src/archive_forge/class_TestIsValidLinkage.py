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
class TestIsValidLinkage:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_linkage_various_size(self, xp):
        for nrow, ncol, valid in [(2, 5, False), (2, 3, False), (1, 4, True), (2, 4, True)]:
            self.check_is_valid_linkage_various_size(nrow, ncol, valid, xp)

    def check_is_valid_linkage_various_size(self, nrow, ncol, valid, xp):
        Z = xp.asarray([[0, 1, 3.0, 2, 5], [3, 2, 4.0, 3, 3]], dtype=xp.float64)
        Z = Z[:nrow, :ncol]
        assert_(is_valid_linkage(Z) == valid)
        if not valid:
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_linkage_int_type(self, xp):
        Z = xp.asarray([[0, 1, 3.0, 2], [3, 2, 4.0, 3]], dtype=xp.int64)
        assert_(is_valid_linkage(Z) is False)
        assert_raises(TypeError, is_valid_linkage, Z, throw=True)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_linkage_empty(self, xp):
        Z = xp.zeros((0, 4), dtype=xp.float64)
        assert_(is_valid_linkage(Z) is False)
        assert_raises(ValueError, is_valid_linkage, Z, throw=True)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_linkage_4_and_up(self, xp):
        for i in range(4, 15, 3):
            y = np.random.rand(i * (i - 1) // 2)
            y = xp.asarray(y)
            Z = linkage(y)
            assert_(is_valid_linkage(Z) is True)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_linkage_4_and_up_neg_index_left(self, xp):
        for i in range(4, 15, 3):
            y = np.random.rand(i * (i - 1) // 2)
            y = xp.asarray(y)
            Z = linkage(y)
            Z[i // 2, 0] = -2
            assert_(is_valid_linkage(Z) is False)
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_linkage_4_and_up_neg_index_right(self, xp):
        for i in range(4, 15, 3):
            y = np.random.rand(i * (i - 1) // 2)
            y = xp.asarray(y)
            Z = linkage(y)
            Z[i // 2, 1] = -2
            assert_(is_valid_linkage(Z) is False)
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_linkage_4_and_up_neg_dist(self, xp):
        for i in range(4, 15, 3):
            y = np.random.rand(i * (i - 1) // 2)
            y = xp.asarray(y)
            Z = linkage(y)
            Z[i // 2, 2] = -0.5
            assert_(is_valid_linkage(Z) is False)
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_linkage_4_and_up_neg_counts(self, xp):
        for i in range(4, 15, 3):
            y = np.random.rand(i * (i - 1) // 2)
            y = xp.asarray(y)
            Z = linkage(y)
            Z[i // 2, 3] = -2
            assert_(is_valid_linkage(Z) is False)
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)