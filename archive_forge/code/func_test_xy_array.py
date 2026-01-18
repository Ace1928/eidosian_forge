from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
@pytest.mark.parametrize('a', all_types)
@pytest.mark.parametrize('func, func_bin', XY_PREDICATES)
def test_xy_array(a, func, func_bin):
    with ignore_invalid(shapely.is_empty(a) and shapely.geos_version < (3, 12, 0)):
        actual = func([a, a], 2, 3)
        expected = func_bin([a, a], Point(2, 3))
    assert actual.shape == (2,)
    assert actual.dtype == np.bool_
    np.testing.assert_allclose(actual, expected)