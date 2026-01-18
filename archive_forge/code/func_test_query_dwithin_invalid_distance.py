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
@pytest.mark.parametrize('geometry,distance,match', [(Point(0, 0), None, 'distance parameter must be provided'), ([Point(0, 0)], None, 'distance parameter must be provided'), (Point(0, 0), 'foo', 'could not convert string to float'), ([Point(0, 0)], 'foo', 'could not convert string to float'), ([Point(0, 0)], ['foo'], 'could not convert string to float'), (Point(0, 0), [0, 1], 'Could not broadcast distance to match geometry'), ([Point(0, 0)], [0, 1], 'Could not broadcast distance to match geometry'), (Point(0, 0), [[1.0]], 'should be one dimensional'), ([Point(0, 0)], [[1.0]], 'should be one dimensional')])
def test_query_dwithin_invalid_distance(tree, geometry, distance, match):
    with pytest.raises(ValueError, match=match):
        tree.query(geometry, predicate='dwithin', distance=distance)