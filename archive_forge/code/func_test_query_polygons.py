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
@pytest.mark.parametrize('geometry,expected', [(Point(0.5, 0.5), [0, 1]), ([Point(0.5, 0.5)], [[0, 0], [0, 1]]), (Point(1, 1), [1]), ([Point(1, 1)], [[0], [1]]), (box(0, 0, 1, 1), [0, 1]), ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]), (box(0, 0, 1.5, 1.5), [0, 1, 2]), ([box(0, 0, 1.5, 1.5)], [[0, 0, 0], [0, 1, 2]]), ([box(0, 0, 1, 1), box(100, 100, 110, 110), box(2, 2, 3, 3)], [[0, 0, 2, 2], [0, 1, 2, 3]]), (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), [2, 3, 4]), ([shapely.buffer(Point(3, 3), HALF_UNIT_DIAG)], [[0, 0, 0], [2, 3, 4]]), (shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG), [1, 2, 3, 4, 5]), ([shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG)], [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]]), (MultiPoint([[5, 7], [7, 5]]), [5, 6, 7]), ([MultiPoint([[5, 7], [7, 5]])], [[0, 0, 0], [5, 6, 7]])])
def test_query_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry), expected)