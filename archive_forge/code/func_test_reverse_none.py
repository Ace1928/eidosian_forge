import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 7, 0), reason='GEOS < 3.7')
def test_reverse_none():
    assert shapely.reverse(None) is None
    assert shapely.reverse([None]).tolist() == [None]
    geometry = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    expected = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    result = shapely.reverse([None, geometry])
    assert result[0] is None
    assert_geometries_equal(result[1], expected)