from math import pi
import pytest
from shapely.geometry import Point
from shapely.wkt import dump, dumps, load, loads
def test_wkt_null(empty_geometry):
    assert empty_geometry.wkt == 'POINT EMPTY'