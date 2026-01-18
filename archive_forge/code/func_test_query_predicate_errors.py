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
@pytest.mark.parametrize('predicate', [pytest.param('within', marks=pytest.mark.xfail(geos_version < (3, 8, 0), reason='GEOS < 3.8')), pytest.param('contains', marks=pytest.mark.xfail(geos_version < (3, 8, 0), reason='GEOS < 3.8')), 'overlaps', 'crosses', 'touches', pytest.param('covers', marks=pytest.mark.xfail(geos_version < (3, 8, 0), reason='GEOS < 3.8')), pytest.param('covered_by', marks=pytest.mark.xfail(geos_version < (3, 8, 0), reason='GEOS < 3.8')), pytest.param('contains_properly', marks=pytest.mark.xfail(geos_version < (3, 8, 0), reason='GEOS < 3.8'))])
def test_query_predicate_errors(tree, predicate):
    with ignore_invalid():
        line_nan = shapely.linestrings([1, 1], [1, float('nan')])
    with pytest.raises(shapely.GEOSException):
        tree.query(line_nan, predicate=predicate)