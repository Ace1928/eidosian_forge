import numpy as np
import pytest
from shapely import Point
from shapely.coords import CoordinateSequence
from shapely.errors import DimensionError
def test_point_immutable():
    p = Point(3.0, 4.0)
    with pytest.raises(AttributeError):
        p.coords = (2.0, 1.0)
    with pytest.raises(TypeError):
        p.coords[0] = (2.0, 1.0)