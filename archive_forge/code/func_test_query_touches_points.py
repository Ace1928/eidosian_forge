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
@pytest.mark.parametrize('geometry,expected', [(Point(0.5, 0.5), []), ([Point(0.5, 0.5)], [[], []]), (Point(1, 1), []), ([Point(1, 1)], [[], []]), (box(3, 3, 6, 6), [3, 6]), ([box(3, 3, 6, 6)], [[0, 0], [3, 6]]), (shapely.buffer(Point(3, 3), 1), []), ([shapely.buffer(Point(3, 3), 1)], [[], []]), (LineString([(-1, -1), (1, 1)]), [1]), ([LineString([(-1, -1), (1, 1)])], [[0], [1]]), (MultiPoint([[5, 5], [7, 7]]), []), ([MultiPoint([[5, 5], [7, 7]])], [[], []])])
def test_query_touches_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate='touches'), expected)