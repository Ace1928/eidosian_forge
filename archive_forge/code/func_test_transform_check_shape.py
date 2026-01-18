import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import shapely
from shapely import count_coordinates, get_coordinates, set_coordinates, transform
from shapely.tests.common import (
def test_transform_check_shape():

    def remove_coord(arr):
        return arr[:-1]
    with pytest.raises(ValueError):
        transform(linear_ring, remove_coord)