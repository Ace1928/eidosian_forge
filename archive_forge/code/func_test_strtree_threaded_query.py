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
def test_strtree_threaded_query():
    polygons = shapely.polygons(np.random.randn(1000, 3, 2))
    N = 100000
    points = shapely.points(4 * np.random.random(N) - 2, 4 * np.random.random(N) - 2)
    n = int(len(polygons) / 4)
    polygons_parts = [polygons[:n], polygons[n:2 * n], polygons[2 * n:3 * n], polygons[3 * n:]]
    n = int(len(points) / 4)
    points_parts = [points[:n], points[n:2 * n], points[2 * n:3 * n], points[3 * n:]]
    trees = []
    for i in range(4):
        left = points_parts[i]
        tree = STRtree(left)
        trees.append(tree)

    def thread_func(idxs):
        i, j = idxs
        tree = trees[i]
        right = polygons_parts[j]
        return tree.query(right, predicate='contains')
    with ThreadPoolExecutor() as pool:
        list(pool.map(thread_func, itertools.product(range(4), range(4))))