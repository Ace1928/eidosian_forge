import unittest
import pytest
from shapely.geometry.polygon import LinearRing, orient, Polygon, signed_area
class PolygonOrienterTestCase(unittest.TestCase):

    def test_no_holes(self):
        ring = LinearRing([(0, 0), (0, 1), (1, 0)])
        polygon = Polygon(ring)
        assert not polygon.exterior.is_ccw
        polygon = orient(polygon, 1)
        assert polygon.exterior.is_ccw

    def test_holes(self):
        polygon = Polygon([(0, 0), (0, 1), (1, 0)], [[(0.5, 0.25), (0.25, 0.5), (0.25, 0.25)]])
        assert not polygon.exterior.is_ccw
        assert polygon.interiors[0].is_ccw
        polygon = orient(polygon, 1)
        assert polygon.exterior.is_ccw
        assert not polygon.interiors[0].is_ccw