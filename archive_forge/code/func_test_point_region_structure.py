import os
import copy
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.spatial._qhull as qhull
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
import itertools
@pytest.mark.parametrize('qhull_opts, extra_pts', [('Qbb Qc Qz', 1), ('Qbb Qc', 0)])
@pytest.mark.parametrize('n_pts', [50, 100])
@pytest.mark.parametrize('ndim', [2, 3])
def test_point_region_structure(self, qhull_opts, n_pts, extra_pts, ndim):
    rng = np.random.default_rng(7790)
    points = rng.random((n_pts, ndim))
    vor = Voronoi(points, qhull_options=qhull_opts)
    pt_region = vor.point_region
    assert pt_region.max() == n_pts - 1 + extra_pts
    assert pt_region.size == len(vor.regions) - extra_pts
    assert len(vor.regions) == n_pts + extra_pts
    assert vor.points.shape[0] == n_pts
    if extra_pts:
        sublens = [len(x) for x in vor.regions]
        assert sublens.count(0) == 1
        assert sublens.index(0) not in pt_region