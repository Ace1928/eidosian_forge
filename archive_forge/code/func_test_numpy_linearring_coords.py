import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_numpy_linearring_coords():
    from numpy.testing import assert_array_equal
    ring = LinearRing([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    ra = np.asarray(ring.coords)
    expected = np.asarray([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)])
    assert_array_equal(ra, expected)