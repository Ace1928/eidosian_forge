import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import (
from shapely.ops import linemerge, split, unary_union
def test_split_multiline_with_point(self):
    l1 = LineString([(0, 1), (2, 1)])
    l2 = LineString([(1, 0), (1, 2)])
    ml = MultiLineString([l1, l2])
    splitter = Point((1, 1))
    self.helper(ml, splitter, 4)