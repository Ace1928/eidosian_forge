import numpy as np
import pytest
from shapely import LineString
class TestCoordsGetItem:

    def test_index_2d_coords(self):
        c = [(float(x), float(-x)) for x in range(4)]
        g = LineString(c)
        for i in range(-4, 4):
            assert g.coords[i] == c[i]
        with pytest.raises(IndexError):
            g.coords[4]
        with pytest.raises(IndexError):
            g.coords[-5]

    def test_index_3d_coords(self):
        c = [(float(x), float(-x), float(x * 2)) for x in range(4)]
        g = LineString(c)
        for i in range(-4, 4):
            assert g.coords[i] == c[i]
        with pytest.raises(IndexError):
            g.coords[4]
        with pytest.raises(IndexError):
            g.coords[-5]

    def test_index_coords_misc(self):
        g = LineString()
        with pytest.raises(IndexError):
            g.coords[0]
        with pytest.raises(TypeError):
            g.coords[0.0]

    def test_slice_2d_coords(self):
        c = [(float(x), float(-x)) for x in range(4)]
        g = LineString(c)
        assert g.coords[1:] == c[1:]
        assert g.coords[:-1] == c[:-1]
        assert g.coords[::-1] == c[::-1]
        assert g.coords[::2] == c[::2]
        assert g.coords[:4] == c[:4]
        assert g.coords[4:] == c[4:] == []

    def test_slice_3d_coords(self):
        c = [(float(x), float(-x), float(x * 2)) for x in range(4)]
        g = LineString(c)
        assert g.coords[1:] == c[1:]
        assert g.coords[:-1] == c[:-1]
        assert g.coords[::-1] == c[::-1]
        assert g.coords[::2] == c[::2]
        assert g.coords[:4] == c[:4]
        assert g.coords[4:] == c[4:] == []