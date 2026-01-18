import json
import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import LineString, Point, shape
from shapely.ops import substring
def test_return_endsubstring_reversed(self):
    assert substring(self.line1, 500, -1).wkt == LineString([(2, 0), (1, 0)]).wkt
    assert substring(self.line3, 4, 2.5).wkt == LineString([(0, 4), (0, 3), (0, 2.5)]).wkt
    assert substring(self.line3, 500, -1.5).wkt == LineString([(0, 4), (0, 3), (0, 2.5)]).wkt
    assert substring(self.line1, 1.1, -0.5, True).wkt == LineString([(2, 0), (1.0, 0)]).wkt
    assert substring(self.line3, 1, 0.5, True).wkt == LineString([(0, 4), (0, 3), (0, 2.0)]).wkt
    assert substring(self.line3, 1.1, -0.5, True).wkt == LineString([(0, 4), (0, 3), (0, 2.0)]).wkt