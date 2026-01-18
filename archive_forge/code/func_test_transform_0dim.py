import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import shapely
from shapely import count_coordinates, get_coordinates, set_coordinates, transform
from shapely.tests.common import (
def test_transform_0dim():
    actual = transform(point, lambda x: x + 1)
    assert isinstance(actual, shapely.Geometry)
    actual = transform(np.asarray(point), lambda x: x + 1)
    assert isinstance(actual, np.ndarray)
    assert actual.ndim == 0