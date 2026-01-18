import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_oriented_evelope_array_like():
    geometries = [Point(1, 1).buffer(1), Point(2, 2).buffer(1)]
    actual = shapely.oriented_envelope(ArrayLike(geometries))
    assert isinstance(actual, ArrayLike)
    expected = shapely.oriented_envelope(geometries)
    assert_geometries_equal(np.asarray(actual), expected)