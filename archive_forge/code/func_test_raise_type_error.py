import json
import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import LineString, Point, shape
from shapely.ops import substring
def test_raise_type_error(self):
    with pytest.raises(GeometryTypeError):
        substring(Point(0, 0), 0, 0)