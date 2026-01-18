import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import (
from shapely.ops import linemerge, split, unary_union
def test_split_line_with_point(self):
    splitter = Point(1, 1)
    self.helper(self.ls, splitter, 2)
    splitter = Point(1.5, 1.5)
    self.helper(self.ls, splitter, 2)
    splitter = Point(3, 4)
    self.helper(self.ls, splitter, 1)
    splitter = Point(2, 2)
    self.helper(self.ls, splitter, 1)