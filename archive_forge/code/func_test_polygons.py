import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
@pytest.mark.parametrize('rings,indices,expected', [([linear_ring, linear_ring], [0, 1], [poly, poly]), ([None, linear_ring], [0, 1], [poly_empty, poly]), ([None, linear_ring, None, None], [0, 0, 1, 1], [poly, poly_empty]), ([linear_ring, hole_1, linear_ring], [0, 0, 1], [poly_hole_1, poly]), ([linear_ring, linear_ring, hole_1], [0, 1, 1], [poly, poly_hole_1]), ([None, linear_ring, linear_ring, hole_1], [0, 0, 1, 1], [poly, poly_hole_1]), ([linear_ring, None, linear_ring, hole_1], [0, 0, 1, 1], [poly, poly_hole_1]), ([linear_ring, None, linear_ring, hole_1], [0, 1, 1, 1], [poly, poly_hole_1]), ([linear_ring, linear_ring, None, hole_1], [0, 1, 1, 1], [poly, poly_hole_1]), ([linear_ring, linear_ring, hole_1, None], [0, 1, 1, 1], [poly, poly_hole_1]), ([linear_ring, hole_1, hole_2, linear_ring], [0, 0, 0, 1], [poly_hole_1_2, poly]), ([linear_ring, hole_1, linear_ring, hole_2], [0, 0, 1, 1], [poly_hole_1, poly_hole_2]), ([linear_ring, linear_ring, hole_1, hole_2], [0, 1, 1, 1], [poly, poly_hole_1_2]), ([linear_ring, hole_1, None, hole_2, linear_ring], [0, 0, 0, 0, 1], [poly_hole_1_2, poly]), ([linear_ring, hole_1, None, linear_ring, hole_2], [0, 0, 0, 1, 1], [poly_hole_1, poly_hole_2]), ([linear_ring, hole_1, linear_ring, None, hole_2], [0, 0, 1, 1, 1], [poly_hole_1, poly_hole_2])])
def test_polygons(rings, indices, expected):
    actual = shapely.polygons(np.array(rings, dtype=object), indices=np.array(indices, dtype=np.intp))
    assert_geometries_equal(actual, expected)