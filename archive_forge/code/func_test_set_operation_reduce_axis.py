import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('func, related_func', REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_axis(func, related_func):
    data = [[point] * 2] * 3
    actual = func(data, axis=None)
    assert isinstance(actual, Geometry)
    actual = func(data, axis=0)
    assert actual.shape == (2,)
    actual = func(data, axis=1)
    assert actual.shape == (3,)
    actual = func(data, axis=-1)
    assert actual.shape == (3,)