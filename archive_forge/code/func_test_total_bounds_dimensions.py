import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
@pytest.mark.parametrize('geom', [point, None, [point, multi_point], [[point, multi_point], [polygon, point]], [[[point, multi_point]], [[polygon, point]]]])
def test_total_bounds_dimensions(geom):
    assert shapely.total_bounds(geom).shape == (4,)