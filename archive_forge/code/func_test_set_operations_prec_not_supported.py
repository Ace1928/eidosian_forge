import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version >= (3, 9, 0), reason='GEOS >= 3.9')
@pytest.mark.parametrize('func', SET_OPERATIONS)
@pytest.mark.parametrize('grid_size', [0, 1])
def test_set_operations_prec_not_supported(func, grid_size):
    with pytest.raises(UnsupportedGEOSVersionError, match='grid_size parameter requires GEOS >= 3.9.0'):
        func(point, point, grid_size)