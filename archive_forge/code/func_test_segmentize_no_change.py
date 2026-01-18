import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='GEOS < 3.10')
@pytest.mark.parametrize('geometry', [point, point_z, multi_point])
def test_segmentize_no_change(geometry):
    actual = shapely.segmentize(geometry, max_segment_length=5)
    assert_geometries_equal(actual, geometry)