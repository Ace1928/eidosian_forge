import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence
def test_numpy_empty_linestring_coords():
    line = LineString([])
    la = np.asarray(line.coords)
    assert la.shape == (0, 2)