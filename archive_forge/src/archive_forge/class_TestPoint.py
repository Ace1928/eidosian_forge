import numpy as np
import pytest
from shapely import Point
from shapely.coords import CoordinateSequence
from shapely.errors import DimensionError
class TestPoint:

    def test_point(self):
        p = Point(1.0, 2.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.coords[:] == [(1.0, 2.0)]
        assert str(p) == p.wkt
        assert p.has_z is False
        with pytest.raises(DimensionError):
            p.z
        p = Point(1.0, 2.0, 3.0)
        assert p.coords[:] == [(1.0, 2.0, 3.0)]
        assert str(p) == p.wkt
        assert p.has_z is True
        assert p.z == 3.0
        p = Point((3.0, 4.0))
        assert p.x == 3.0
        assert p.y == 4.0
        assert tuple(p.coords) == ((3.0, 4.0),)
        assert p.coords[0] == (3.0, 4.0)
        with pytest.raises(IndexError):
            p.coords[1]
        assert p.bounds == (3.0, 4.0, 3.0, 4.0)
        assert p.__geo_interface__ == {'type': 'Point', 'coordinates': (3.0, 4.0)}

    def test_point_empty(self):
        p_null = Point()
        assert p_null.wkt == 'POINT EMPTY'
        assert p_null.coords[:] == []
        assert p_null.area == 0.0

    def test_coords(self):
        p = Point(0.0, 0.0, 1.0)
        coords = p.coords[0]
        assert coords == (0.0, 0.0, 1.0)
        a = np.asarray(coords)
        assert a.ndim == 1
        assert a.size == 3
        assert a.shape == (3,)