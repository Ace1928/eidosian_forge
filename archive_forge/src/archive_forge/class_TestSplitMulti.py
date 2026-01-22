import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import (
from shapely.ops import linemerge, split, unary_union
class TestSplitMulti(TestSplitGeometry):

    def test_split_multiline_with_point(self):
        l1 = LineString([(0, 1), (2, 1)])
        l2 = LineString([(1, 0), (1, 2)])
        ml = MultiLineString([l1, l2])
        splitter = Point((1, 1))
        self.helper(ml, splitter, 4)

    def test_split_multiline_with_multipoint(self):
        l1 = LineString([(0, 1), (3, 1)])
        l2 = LineString([(1, 0), (1, 2)])
        ml = MultiLineString([l1, l2])
        splitter = MultiPoint([(1, 1), (2, 1), (4, 2)])
        self.helper(ml, splitter, 5)

    def test_split_multipolygon_with_line(self):
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        poly2 = Polygon([(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)])
        mpoly = MultiPolygon([poly1, poly2])
        ls = LineString([(-1, -1), (3, 3)])
        self.helper(mpoly, ls, 4)
        poly1 = Polygon([(10, 10), (10, 11), (11, 11), (11, 10), (10, 10)])
        poly2 = Polygon([(-10, -10), (-10, -11), (-11, -11), (-11, -10), (-10, -10)])
        mpoly = MultiPolygon([poly1, poly2])
        ls = LineString([(-1, -1), (3, 3)])
        self.helper(mpoly, ls, 2)