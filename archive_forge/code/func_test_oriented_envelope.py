import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('func', [shapely.oriented_envelope, shapely.minimum_rotated_rectangle])
@pytest.mark.parametrize('geometry, expected', [(MultiPoint([(1.0, 1.0), (1.0, 5.0), (3.0, 6.0), (4.0, 2.0), (5.0, 5.0)]), Polygon([(1.0, 1.0), (1.0, 6.0), (5.0, 6.0), (5.0, 1.0), (1.0, 1.0)])), (LineString([(1, 1), (5, 1), (10, 10)]), Polygon([(1, 1), (3, -1), (12, 8), (10, 10), (1, 1)])), (Polygon([(1, 1), (15, 1), (5, 9), (1, 1)]), Polygon([(1.0, 1.0), (5.0, 9.0), (16.2, 3.4), (12.2, -4.6), (1.0, 1.0)])), (LineString([(1, 1), (10, 1)]), LineString([(1, 1), (10, 1)])), (Point(2, 2), Point(2, 2)), (GeometryCollection(), Polygon())])
def test_oriented_envelope(geometry, expected, func):
    actual = func(geometry)
    assert_geometries_equal(actual, expected, normalize=True, tolerance=0.001)