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
@pytest.mark.parametrize('geometry,expected', [(Point(0, 0), []), ([Point(0, 0)], [[], []]), (box(0, 0, 1.5, 1.5), [1]), ([box(0, 0, 1.5, 1.5)], [[0], [1]]), (shapely.buffer(Point(3, 3), 0.5), [2, 3]), ([shapely.buffer(Point(3, 3), 0.5)], [[0, 0], [2, 3]]), (shapely.linestrings([(1, 0), (0, 1)]), [0]), ([shapely.linestrings([(1, 0), (0, 1)])], [[0], [0]]), (MultiPoint([[5, 7], [7, 5]]), []), ([MultiPoint([[5, 7], [7, 5]])], [[], []]), (MultiPoint([[5, 7], [7, 7], [7, 8]]), []), ([MultiPoint([[5, 7], [7, 7], [7, 8]])], [[], []])])
def test_query_crosses_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate='crosses'), expected)