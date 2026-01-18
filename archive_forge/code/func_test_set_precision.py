import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 6, 0), reason='GEOS < 3.6')
@pytest.mark.parametrize('mode', ('valid_output', 'pointwise', 'keep_collapsed'))
def test_set_precision(mode):
    initial_geometry = Point(0.9, 0.9)
    assert shapely.get_precision(initial_geometry) == 0
    geometry = shapely.set_precision(initial_geometry, 0, mode=mode)
    assert shapely.get_precision(geometry) == 0
    assert_geometries_equal(geometry, initial_geometry)
    geometry = shapely.set_precision(initial_geometry, 1, mode=mode)
    assert shapely.get_precision(geometry) == 1
    assert_geometries_equal(geometry, Point(1, 1))
    assert_geometries_equal(initial_geometry, Point(0.9, 0.9))