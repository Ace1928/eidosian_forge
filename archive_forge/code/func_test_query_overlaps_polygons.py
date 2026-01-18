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
@pytest.mark.parametrize('geometry,expected', [(Point(0, 0), []), ([Point(0, 0)], [[], []]), (box(0, 0, 1, 1), [0, 1]), ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]), (box(0, 0, 2, 2), [0, 2]), ([box(0, 0, 2, 2)], [[0, 0], [0, 2]]), (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), [2, 4]), ([shapely.buffer(Point(3, 3), HALF_UNIT_DIAG)], [[0, 0], [2, 4]]), (shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG), [1, 2, 4, 5]), ([shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG)], [[0, 0, 0, 0], [1, 2, 4, 5]]), (MultiPoint([[5, 7], [7, 7]]), []), ([MultiPoint([[5, 7], [7, 7]])], [[], []])])
def test_query_overlaps_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate='overlaps'), expected)