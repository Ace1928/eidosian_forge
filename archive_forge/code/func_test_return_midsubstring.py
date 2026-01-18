import json
import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import LineString, Point, shape
from shapely.ops import substring
def test_return_midsubstring(self):
    assert substring(self.line1, 0.5, 0.6).wkt == LineString([(0.5, 0), (0.6, 0)]).wkt
    assert substring(self.line1, -0.6, -0.5).wkt == LineString([(1.4, 0), (1.5, 0)]).wkt
    assert substring(self.line1, 0.5, 0.6, True).wkt == LineString([(1, 0), (1.2, 0)]).wkt
    assert substring(self.line1, -0.6, -0.5, True).wkt == LineString([(0.8, 0), (1, 0)]).wkt