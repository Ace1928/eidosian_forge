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
class TestMaxDists:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_maxdists_empty_linkage(self, xp):
        Z = xp.zeros((0, 4), dtype=xp.float64)
        assert_raises(ValueError, maxdists, Z)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_maxdists_one_cluster_linkage(self, xp):
        Z = xp.asarray([[0, 1, 0.3, 4]], dtype=xp.float64)
        MD = maxdists(Z)
        expectedMD = calculate_maximum_distances(Z, xp)
        xp_assert_close(MD, expectedMD, atol=1e-15)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_maxdists_Q_linkage(self, xp):
        for method in ['single', 'complete', 'ward', 'centroid', 'median']:
            self.check_maxdists_Q_linkage(method, xp)

    def check_maxdists_Q_linkage(self, method, xp):
        X = xp.asarray(hierarchy_test_data.Q_X)
        Z = linkage(X, method)
        MD = maxdists(Z)
        expectedMD = calculate_maximum_distances(Z, xp)
        xp_assert_close(MD, expectedMD, atol=1e-15)