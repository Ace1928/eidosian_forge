import numpy as np
import pytest
from shapely.geometry import Point, Polygon
from shapely.prepared import prep, PreparedGeometry
def test_prepared_predicates():
    polygon1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    polygon2 = Polygon([(0.5, 0.5), (1.5, 0.5), (1.0, 1.0), (0.5, 0.5)])
    point2 = Point(0.5, 0.5)
    polygon_empty = Polygon()
    prepared_polygon1 = PreparedGeometry(polygon1)
    for geom2 in (polygon2, point2, polygon_empty):
        with np.errstate(invalid='ignore'):
            assert polygon1.disjoint(geom2) == prepared_polygon1.disjoint(geom2)
            assert polygon1.touches(geom2) == prepared_polygon1.touches(geom2)
            assert polygon1.intersects(geom2) == prepared_polygon1.intersects(geom2)
            assert polygon1.crosses(geom2) == prepared_polygon1.crosses(geom2)
            assert polygon1.within(geom2) == prepared_polygon1.within(geom2)
            assert polygon1.contains(geom2) == prepared_polygon1.contains(geom2)
            assert polygon1.overlaps(geom2) == prepared_polygon1.overlaps(geom2)