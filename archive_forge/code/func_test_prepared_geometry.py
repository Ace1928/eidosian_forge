import numpy as np
import pytest
from shapely.geometry import Point, Polygon
from shapely.prepared import prep, PreparedGeometry
def test_prepared_geometry():
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p = PreparedGeometry(polygon)
    assert p.contains(Point(0.5, 0.5))
    assert not p.contains(Point(0.5, 1.5))