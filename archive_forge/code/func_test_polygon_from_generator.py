import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_polygon_from_generator():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    gen = (coord for coord in coords)
    polygon = Polygon(gen)
    assert polygon.exterior.coords[:] == coords