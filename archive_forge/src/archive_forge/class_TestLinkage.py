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
class TestLinkage:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_linkage_non_finite_elements_in_distance_matrix(self, xp):
        y = xp.zeros((6,))
        y[0] = xp.nan
        assert_raises(ValueError, linkage, y)

    def test_linkage_empty_distance_matrix(self):
        y = np.zeros((0,))
        assert_raises(ValueError, linkage, y)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_linkage_tdist(self, xp):
        for method in ['single', 'complete', 'average', 'weighted']:
            self.check_linkage_tdist(method, xp)

    def check_linkage_tdist(self, method, xp):
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), method)
        expectedZ = getattr(hierarchy_test_data, 'linkage_ytdist_' + method)
        xp_assert_close(Z, xp.asarray(expectedZ), atol=1e-10)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_linkage_X(self, xp):
        for method in ['centroid', 'median', 'ward']:
            self.check_linkage_q(method, xp)

    def check_linkage_q(self, method, xp):
        Z = linkage(xp.asarray(hierarchy_test_data.X), method)
        expectedZ = getattr(hierarchy_test_data, 'linkage_X_' + method)
        xp_assert_close(Z, xp.asarray(expectedZ), atol=1e-06)
        y = scipy.spatial.distance.pdist(hierarchy_test_data.X, metric='euclidean')
        Z = linkage(xp.asarray(y), method)
        xp_assert_close(Z, xp.asarray(expectedZ), atol=1e-06)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_compare_with_trivial(self, xp):
        rng = np.random.RandomState(0)
        n = 20
        X = rng.rand(n, 2)
        d = pdist(X)
        for method, code in _LINKAGE_METHODS.items():
            Z_trivial = _hierarchy.linkage(d, n, code)
            Z = linkage(xp.asarray(d), method)
            xp_assert_close(Z, xp.asarray(Z_trivial), rtol=1e-14, atol=1e-15)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_optimal_leaf_ordering(self, xp):
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), optimal_ordering=True)
        expectedZ = getattr(hierarchy_test_data, 'linkage_ytdist_single_olo')
        xp_assert_close(Z, xp.asarray(expectedZ), atol=1e-10)