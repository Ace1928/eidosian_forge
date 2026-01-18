import itertools
import math
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from numpy.testing import assert_array_equal
import shapely
from shapely import box, geos_version, LineString, MultiPoint, Point, STRtree
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
import pickle
import sys
from shapely import Point, geos_version
@pytest.mark.parametrize('geometry,expected', [(Point(0, 0), ([0], [0.0])), ([Point(0, 0)], ([[0], [0]], [0.0])), (Point(0.5, 0.5), ([0, 1], [0.7071, 0.7071])), ([Point(0.5, 0.5)], ([[0, 0], [0, 1]], [0.7071, 0.7071])), (box(0, 0, 1, 1), ([0, 1], [0.0, 0.0])), ([box(0, 0, 1, 1)], ([[0, 0], [0, 1]], [0.0, 0.0]))])
def test_query_nearest_return_distance(tree, geometry, expected):
    expected_indices, expected_dist = expected
    actual_indices, actual_dist = tree.query_nearest(geometry, return_distance=True)
    assert_array_equal(actual_indices, expected_indices)
    assert_array_equal(np.round(actual_dist, 4), expected_dist)