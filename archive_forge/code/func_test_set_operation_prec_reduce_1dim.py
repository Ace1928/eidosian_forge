import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 9, 0), reason='GEOS < 3.9')
@pytest.mark.parametrize('n', range(1, 5))
@pytest.mark.parametrize('func, related_func', REDUCE_SET_OPERATIONS_PREC)
@pytest.mark.parametrize('grid_size', [0, 1])
def test_set_operation_prec_reduce_1dim(n, func, related_func, grid_size):
    actual = func(reduce_test_data[:n], grid_size=grid_size)
    expected = reduce_test_data[0]
    for i in range(1, n):
        expected = related_func(expected, reduce_test_data[i], grid_size=grid_size)
    assert shapely.equals(actual, expected)