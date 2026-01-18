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
def test_query_with_partially_prepared_inputs(tree):
    geom = np.array([box(0, 0, 1, 1), box(3, 3, 5, 5)])
    expected = tree.query(geom, predicate='intersects')
    shapely.prepare(geom[0])
    assert_array_equal(expected, tree.query(geom, predicate='intersects'))