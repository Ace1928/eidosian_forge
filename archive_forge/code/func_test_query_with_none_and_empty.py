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
@pytest.mark.parametrize('tree_geometry, geometry,expected', [([], point, []), ([], [point], [[], []]), ([], None, []), ([], [None], [[], []]), ([None], point, []), ([None], [point], [[], []]), ([None], None, []), ([None], [None], [[], []]), ([point], None, []), ([point], [None], [[], []]), ([empty], empty, []), ([empty], [empty], [[], []]), ([empty], point, []), ([empty], [point], [[], []]), ([point, empty], empty, []), ([point, empty], [empty], [[], []]), ([None, point], box(0, 0, 10, 10), [1]), ([None, point], [box(0, 0, 10, 10)], [[0], [1]]), ([None, empty, point], box(0, 0, 10, 10), [2]), ([point, None, point], box(0, 0, 10, 10), [0, 2]), ([point, None, point], [box(0, 0, 10, 10)], [[0, 0], [0, 2]]), ([empty, point], [empty, point], [[1], [1]]), ([empty, empty_point, empty_line_string, point], [empty, empty_point, empty_line_string, point], [[3], [3]])])
def test_query_with_none_and_empty(tree_geometry, geometry, expected):
    tree = STRtree(tree_geometry)
    assert_array_equal(tree.query(geometry), expected)