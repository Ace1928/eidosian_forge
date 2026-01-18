import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import shapely
from shapely import count_coordinates, get_coordinates, set_coordinates, transform
from shapely.tests.common import (
def test_set_coords_breaks_ring():
    with pytest.raises(shapely.GEOSException):
        set_coordinates(linear_ring, np.random.random((5, 2)))