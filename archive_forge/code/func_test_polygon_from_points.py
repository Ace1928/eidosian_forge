import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_polygon_from_points():
    polygon = Polygon([Point(0.0, 0.0), Point(0.0, 1.0), Point(1.0, 1.0)])
    expected_coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]
    assert polygon.exterior.coords[:] == expected_coords