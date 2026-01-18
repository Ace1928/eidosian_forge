import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import (
from shapely.ops import linemerge, split, unary_union
def test_split_line_with_multiline(self):
    splitter = MultiLineString([[(0, 1), (1, 0)], [(0, 0), (2, -2)]])
    self.helper(self.ls, splitter, 2)
    splitter = MultiLineString([[(0, 1), (1, 0)], [(0, 2), (2, 0)]])
    self.helper(self.ls, splitter, 3)
    splitter = MultiLineString([[(0, 1), (1, 0)], [(0, 2), (2, 0), (2.2, 3.2)]])
    self.helper(self.ls, splitter, 4)
    splitter = MultiLineString([[(0, 0), (1.5, 1.5)], [(1.5, 1.5), (3, 4)]])
    with pytest.raises(ValueError):
        self.helper(self.ls, splitter, 1)
    splitter = MultiLineString([[(0, 1), (0, 2)], [(1, 0), (2, 0)]])
    self.helper(self.ls, splitter, 1)