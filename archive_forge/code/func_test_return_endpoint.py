import json
import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import LineString, Point, shape
from shapely.ops import substring
def test_return_endpoint(self):
    assert substring(self.line1, 500, 600).equals(Point(2, 0))
    assert substring(self.line1, 500, 500).equals(Point(2, 0))
    assert substring(self.line1, 1, 1.1, True).equals(Point(2, 0))
    assert substring(self.line1, 1.1, 1.1, True).equals(Point(2, 0))