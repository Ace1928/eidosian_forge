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
@pytest.mark.skipif(geos_version < (3, 10, 0), reason='GEOS < 3.10')
@pytest.mark.parametrize('geometry,distance,expected', [(None, 1.0, []), ([None], 1.0, [[], []]), (Point(0.25, 0.25), 0, []), ([Point(0.25, 0.25)], 0, [[], []]), (Point(0.25, 0.25), -1, []), ([Point(0.25, 0.25)], -1, [[], []]), (Point(0.25, 0.25), np.nan, []), ([Point(0.25, 0.25)], np.nan, [[], []]), (Point(), 1, []), ([Point()], 1, [[], []]), (Point(0.25, 0.25), 0.5, [0]), ([Point(0.25, 0.25)], 0.5, [[0], [0]]), (Point(0.25, 0.25), 2.5, [0, 1, 2]), ([Point(0.25, 0.25)], 2.5, [[0, 0, 0], [0, 1, 2]]), (Point(3, 3), 1.5, [2, 3, 4]), ([Point(3, 3)], 1.5, [[0, 0, 0], [2, 3, 4]]), (Point(0.5, 0.5), 0.75, [0, 1]), ([Point(0.5, 0.5)], 0.75, [[0, 0], [0, 1]]), ([None, Point(0.5, 0.5)], 0.75, [[1, 1], [0, 1]]), ([Point(0.5, 0.5), Point(0.25, 0.25)], 0.75, [[0, 0, 1], [0, 1, 0]]), ([Point(0, 0.2), Point(1.75, 1.75)], [0.25, 2], [[0, 1, 1, 1], [0, 1, 2, 3]]), (box(0, 0, 3, 3), 0, [0, 1, 2, 3]), ([box(0, 0, 3, 3)], 0, [[0, 0, 0, 0], [0, 1, 2, 3]]), (box(0, 0, 3, 3), 0.25, [0, 1, 2, 3]), ([box(0, 0, 3, 3)], 0.25, [[0, 0, 0, 0], [0, 1, 2, 3]]), (box(1, 1, 2, 2), 1.5, [0, 1, 2, 3]), ([box(1, 1, 2, 2)], 1.5, [[0, 0, 0, 0], [0, 1, 2, 3]]), (MultiPoint([[0.25, 0.25], [1.5, 1.5]]), 0.75, [0, 1, 2]), ([MultiPoint([[0.25, 0.25], [1.5, 1.5]])], 0.75, [[0, 0, 0], [0, 1, 2]]), (MultiPoint([[0.5, 0.5], [3.5, 3.5]]), 0.75, [0, 1, 3, 4]), ([MultiPoint([[0.5, 0.5], [3.5, 3.5]])], 0.75, [[0, 0, 0, 0], [0, 1, 3, 4]])])
def test_query_dwithin_points(tree, geometry, distance, expected):
    assert_array_equal(tree.query(geometry, predicate='dwithin', distance=distance), expected)