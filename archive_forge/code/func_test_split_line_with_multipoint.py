import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import (
from shapely.ops import linemerge, split, unary_union
def test_split_line_with_multipoint(self):
    splitter = MultiPoint([(1, 1), (1.5, 1.5), (0.5, 0.5)])
    self.helper(self.ls, splitter, 4)
    splitter = MultiPoint([(1, 1), (3, 4)])
    self.helper(self.ls, splitter, 2)
    splitter = MultiPoint([(1, 1), (1.5, 1.5), (1, 1)])
    self.helper(self.ls, splitter, 3)