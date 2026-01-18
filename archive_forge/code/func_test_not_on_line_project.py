import unittest
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, Point
def test_not_on_line_project(self):
    assert self.line1.project(Point(-10, -10)) == 0.0