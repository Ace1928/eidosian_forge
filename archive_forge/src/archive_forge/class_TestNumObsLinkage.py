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
class TestNumObsLinkage:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_num_obs_linkage_empty(self, xp):
        Z = xp.zeros((0, 4), dtype=xp.float64)
        assert_raises(ValueError, num_obs_linkage, Z)

    @array_api_compatible
    def test_num_obs_linkage_1x4(self, xp):
        Z = xp.asarray([[0, 1, 3.0, 2]], dtype=xp.float64)
        assert_equal(num_obs_linkage(Z), 2)

    @array_api_compatible
    def test_num_obs_linkage_2x4(self, xp):
        Z = xp.asarray([[0, 1, 3.0, 2], [3, 2, 4.0, 3]], dtype=xp.float64)
        assert_equal(num_obs_linkage(Z), 3)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_num_obs_linkage_4_and_up(self, xp):
        for i in range(4, 15, 3):
            y = np.random.rand(i * (i - 1) // 2)
            y = xp.asarray(y)
            Z = linkage(y)
            assert_equal(num_obs_linkage(Z), i)