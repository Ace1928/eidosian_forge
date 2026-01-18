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
def test_query_nearest_nonscalar_max_distance(tree):
    with pytest.raises(ValueError, match='parameter only accepts scalar values'):
        tree.query_nearest(Point(0.5, 0.5), max_distance=[1])