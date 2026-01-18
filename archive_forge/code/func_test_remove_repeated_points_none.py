import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 11, 0), reason='GEOS < 3.11')
def test_remove_repeated_points_none():
    assert shapely.remove_repeated_points(None, 1) is None
    assert shapely.remove_repeated_points([None], 1).tolist() == [None]
    geometry = LineString([(0, 0), (0, 0), (1, 1)])
    expected = LineString([(0, 0), (1, 1)])
    result = shapely.remove_repeated_points([None, geometry], 1)
    assert result[0] is None
    assert_geometries_equal(result[1], expected)