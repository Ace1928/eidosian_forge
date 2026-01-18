import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import (
from shapely.ops import linemerge, split, unary_union
def test_split_line_with_polygon(self):
    splitter = Polygon([(1, 0), (1, 2), (2, 2), (2, 0), (1, 0)])
    self.helper(self.ls, splitter, 3)
    splitter = Polygon([(0, 0), (1, 2), (2, 2), (1, 0), (0, 0)])
    self.helper(self.ls, splitter, 2)
    splitter = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)], [[(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5), (0.5, 0.5)]])
    self.helper(self.ls, splitter, 4)