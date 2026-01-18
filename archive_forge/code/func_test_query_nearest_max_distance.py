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
@pytest.mark.parametrize('geometry,max_distance,expected', [(Point(0.5, 0.5), None, [0, 1]), ([Point(0.5, 0.5)], None, [[0, 0], [0, 1]]), (Point(0.5, 0.5), 10, [0, 1]), ([Point(0.5, 0.5)], 10, [[0, 0], [0, 1]]), (Point(0.5, 0.5), 0.1, []), ([Point(0.5, 0.5)], 0.1, [[], []]), ([Point(0.5, 0.5), Point(0, 0)], 0.1, [[1], [0]])])
def test_query_nearest_max_distance(tree, geometry, max_distance, expected):
    assert_array_equal(tree.query_nearest(geometry, max_distance=max_distance), expected)