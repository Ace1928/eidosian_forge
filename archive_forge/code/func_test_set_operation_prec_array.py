import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 9, 0), reason='GEOS < 3.9')
@pytest.mark.parametrize('a', all_single_types)
@pytest.mark.parametrize('func', SET_OPERATIONS)
@pytest.mark.parametrize('grid_size', [0, 1, 2])
def test_set_operation_prec_array(a, func, grid_size):
    actual = func([a, a], point, grid_size=grid_size)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)
    b = shapely.set_precision(a, grid_size=grid_size)
    point2 = shapely.set_precision(point, grid_size=grid_size)
    expected = func([b, b], point2)
    assert shapely.equals(shapely.normalize(actual), shapely.normalize(expected)).all()