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
def test_pathological(self):
    points = DATASETS['pathological-1']
    tri = qhull.Delaunay(points)
    assert_equal(tri.points[tri.simplices].max(), points.max())
    assert_equal(tri.points[tri.simplices].min(), points.min())
    points = DATASETS['pathological-2']
    tri = qhull.Delaunay(points)
    assert_equal(tri.points[tri.simplices].max(), points.max())
    assert_equal(tri.points[tri.simplices].min(), points.min())