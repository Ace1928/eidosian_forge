import numpy as np
import pytest
from shapely import LineString
def test_slice_3d_coords(self):
    c = [(float(x), float(-x), float(x * 2)) for x in range(4)]
    g = LineString(c)
    assert g.coords[1:] == c[1:]
    assert g.coords[:-1] == c[:-1]
    assert g.coords[::-1] == c[::-1]
    assert g.coords[::2] == c[::2]
    assert g.coords[:4] == c[:4]
    assert g.coords[4:] == c[4:] == []