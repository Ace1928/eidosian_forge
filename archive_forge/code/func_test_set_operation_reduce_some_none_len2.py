import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('func, related_func', REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_some_none_len2(func, related_func):
    assert func([empty, None]) == empty