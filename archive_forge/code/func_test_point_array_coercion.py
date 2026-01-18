import numpy as np
import pytest
from shapely import Point
from shapely.coords import CoordinateSequence
from shapely.errors import DimensionError
def test_point_array_coercion():
    p = Point(3.0, 4.0)
    arr = np.array(p)
    assert arr.ndim == 0
    assert arr.size == 1
    assert arr.dtype == np.dtype('object')
    assert arr.item() == p