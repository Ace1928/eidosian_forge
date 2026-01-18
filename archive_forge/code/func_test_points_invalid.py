import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
def test_points_invalid():
    with pytest.raises(shapely.GEOSException):
        shapely.points([[1, 1], [2, 2]], indices=[0, 0])