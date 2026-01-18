import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
@pytest.mark.parametrize('indices,expected', [([0, 1], [point, point, empty_point, None]), ([0, 3], [point, None, empty_point, point]), ([2, 3], [None, None, point, point])])
def test_points_out(indices, expected):
    out = np.empty(4, dtype=object)
    out[2] = empty_point
    actual = shapely.points([[2, 3], [2, 3]], indices=indices, out=out)
    assert_geometries_equal(out, expected)
    assert actual is out