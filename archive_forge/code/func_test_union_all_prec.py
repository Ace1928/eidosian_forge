import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 9, 0), reason='GEOS < 3.9')
@pytest.mark.parametrize('geom,grid_size,expected', [([shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)], 0, Polygon(((0, 0.2), (0, 10), (5.1, 10), (5.1, 0.2), (5, 0.2), (5, 0.1), (0.1, 0.1), (0.1, 0.2), (0, 0.2)))), ([shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)], 0.1, Polygon(((0, 0.2), (0, 10), (5.1, 10), (5.1, 0.2), (5, 0.2), (5, 0.1), (0.1, 0.1), (0.1, 0.2), (0, 0.2)))), ([shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)], 1, Polygon([(0, 5), (0, 10), (5, 10), (5, 5), (5, 0), (0, 0), (0, 5)])), ([shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)], 10, Polygon([(0, 10), (10, 10), (10, 0), (0, 0), (0, 10)])), ([shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)], 100, Polygon())])
def test_union_all_prec(geom, grid_size, expected):
    actual = shapely.union_all(geom, grid_size=grid_size)
    assert shapely.equals(actual, expected)