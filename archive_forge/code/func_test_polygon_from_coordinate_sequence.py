import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_polygon_from_coordinate_sequence():
    coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]
    polygon = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    assert polygon.exterior.coords[:] == coords
    assert len(polygon.interiors) == 0
    polygon = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    assert polygon.exterior.coords[:] == coords
    assert len(polygon.interiors) == 0