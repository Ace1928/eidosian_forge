import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version >= (3, 12, 0) or shapely.geos_version < (3, 8, 0), reason='GEOS >= 3.12')
@pytest.mark.parametrize('geometry, expected', [(MultiPoint([(1.0, 1.0), (1.0, 5.0), (3.0, 6.0), (4.0, 2.0), (5.0, 5.0)]), Polygon([(-0.2, 1.4), (1.5, 6.5), (5.1, 5.3), (3.4, 0.2), (-0.2, 1.4)])), (LineString([(1, 1), (5, 1), (10, 10)]), Polygon([(1, 1), (3, -1), (12, 8), (10, 10), (1, 1)])), (Polygon([(1, 1), (15, 1), (5, 9), (1, 1)]), Polygon([(1.0, 1.0), (1.0, 9.0), (15.0, 9.0), (15.0, 1.0), (1.0, 1.0)])), (LineString([(1, 1), (10, 1)]), LineString([(1, 1), (10, 1)])), (Point(2, 2), Point(2, 2)), (GeometryCollection(), Polygon())])
def test_oriented_envelope_pre_geos_312(geometry, expected):
    actual = shapely.constructive._oriented_envelope_geos(geometry)
    if shapely.geos_version < (3, 8, 0):
        assert shapely.equals(actual, expected).all()
    else:
        assert_geometries_equal(actual, expected, normalize=True, tolerance=0.001)