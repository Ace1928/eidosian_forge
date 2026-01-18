import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence
def test_numpy_linestring_coords(self):
    from numpy.testing import assert_array_equal
    line = LineString([(1.0, 2.0), (3.0, 4.0)])
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    la = np.asarray(line.coords)
    assert_array_equal(la, expected)