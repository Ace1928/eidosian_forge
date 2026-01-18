import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_polygon_from_polygon():
    coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
    polygon = Polygon(coords, [[(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25)]])
    copy = Polygon(polygon)
    assert len(copy.exterior.coords) == 5
    assert len(copy.interiors) == 1
    assert len(copy.interiors[0].coords) == 5