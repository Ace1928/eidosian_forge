import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('none_position', range(3))
@pytest.mark.parametrize('func, related_func', REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_two_none(func, related_func, none_position):
    test_data = reduce_test_data[:2]
    test_data.insert(none_position, None)
    test_data.insert(none_position, None)
    actual = func(test_data)
    expected = related_func(reduce_test_data[0], reduce_test_data[1])
    assert_geometries_equal(actual, expected)