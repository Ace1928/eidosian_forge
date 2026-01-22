import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import (
from shapely.ops import linemerge, split, unary_union
class TestSplitPolygon(TestSplitGeometry):
    poly_simple = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    poly_hole = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)], [[(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5), (0.5, 0.5)]])

    def test_split_poly_with_line(self):
        splitter = LineString([(1, 3), (1, -3)])
        self.helper(self.poly_simple, splitter, 2)
        self.helper(self.poly_hole, splitter, 2)
        splitter = LineString([(0, 2), (5, 2)])
        self.helper(self.poly_simple, splitter, 1)
        self.helper(self.poly_hole, splitter, 1)
        splitter = LineString([(0.2, 0.2), (1.7, 1.7), (3, 2)])
        self.helper(self.poly_simple, splitter, 1)
        self.helper(self.poly_hole, splitter, 1)
        splitter = LineString([(0, 3), (3, 3), (3, 0)])
        self.helper(self.poly_simple, splitter, 1)
        self.helper(self.poly_hole, splitter, 1)

    def test_split_poly_with_other(self):
        with pytest.raises(GeometryTypeError):
            split(self.poly_simple, Point(1, 1))
        with pytest.raises(GeometryTypeError):
            split(self.poly_simple, MultiPoint([(1, 1), (3, 4)]))
        with pytest.raises(GeometryTypeError):
            split(self.poly_simple, self.poly_hole)