import json
import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import LineString, Point, shape
from shapely.ops import substring
def test_return_substring_with_vertices(self):
    assert substring(self.line2, 1, 7).wkt == LineString([(3, 1), (3, 6), (4, 6)]).wkt
    assert substring(self.line2, 0.2, 0.9, True).wkt == LineString([(3, 1.5), (3, 6), (3.75, 6)]).wkt
    assert substring(self.line2, 0, 0.9, True).wkt == LineString([(3, 0), (3, 6), (3.75, 6)]).wkt
    assert substring(self.line2, 0.2, 1, True).wkt == LineString([(3, 1.5), (3, 6), (4.5, 6)]).wkt