import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import shapely
from shapely import count_coordinates, get_coordinates, set_coordinates, transform
from shapely.tests.common import (
@pytest.mark.parametrize('geom', [pytest.param(empty_point_z, marks=pytest.mark.skipif(shapely.geos_version < (3, 9, 0), reason="Empty points don't have a dimensionality before GEOS 3.9")), empty_line_string_z])
def test_transform_empty_preserve_z(geom):
    assert shapely.get_coordinate_dimension(geom) == 3
    new_geom = transform(geom, lambda x: x + 1, include_z=True)
    assert shapely.get_coordinate_dimension(new_geom) == 3