import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import (
from shapely.ops import linemerge, split, unary_union
def test_split_closed_line_with_point(self):
    ls = LineString([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    splitter = Point(0, 0)
    self.helper(ls, splitter, 1)