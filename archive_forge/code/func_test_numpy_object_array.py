import numpy as np
import pytest
from shapely import MultiPolygon, Polygon
from shapely.geometry.base import dump_coords
from shapely.tests.geometry.test_multi import MultiGeometryTestCase
def test_numpy_object_array():
    geom = MultiPolygon([(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)), [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))])])
    ar = np.empty(1, object)
    ar[:] = [geom]
    assert ar[0] == geom