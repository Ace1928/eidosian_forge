import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 9, 0), reason='GEOS < 3.9')
@pytest.mark.parametrize('func, related_func', REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_axis(func, related_func):
    data = [[point] * 2] * 3
    actual = func(data, grid_size=1, axis=None)
    assert isinstance(actual, Geometry)
    actual = func(data, grid_size=1, axis=0)
    assert actual.shape == (2,)
    actual = func(data, grid_size=1, axis=1)
    assert actual.shape == (3,)
    actual = func(data, grid_size=1, axis=-1)
    assert actual.shape == (3,)