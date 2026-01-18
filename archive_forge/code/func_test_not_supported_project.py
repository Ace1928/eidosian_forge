import unittest
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, Point
def test_not_supported_project(self):
    with pytest.raises(shapely.GEOSException, match='IllegalArgumentException'):
        self.point.buffer(1.0).project(self.point)