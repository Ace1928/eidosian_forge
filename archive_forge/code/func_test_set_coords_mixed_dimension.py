import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import shapely
from shapely import count_coordinates, get_coordinates, set_coordinates, transform
from shapely.tests.common import (
@pytest.mark.parametrize('include_z', [True, False])
def test_set_coords_mixed_dimension(include_z):
    geoms = np.array([point, point_z], dtype=object)
    coords = get_coordinates(geoms, include_z=include_z)
    new_geoms = set_coordinates(geoms, coords * 2)
    if include_z:
        assert not shapely.has_z(new_geoms[0])
        assert shapely.has_z(new_geoms[1])
    else:
        assert not shapely.has_z(new_geoms).any()