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
class TestInconsistent:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_inconsistent_tdist(self, xp):
        for depth in hierarchy_test_data.inconsistent_ytdist:
            self.check_inconsistent_tdist(depth, xp)

    def check_inconsistent_tdist(self, depth, xp):
        Z = xp.asarray(hierarchy_test_data.linkage_ytdist_single)
        xp_assert_close(inconsistent(Z, depth), xp.asarray(hierarchy_test_data.inconsistent_ytdist[depth]))