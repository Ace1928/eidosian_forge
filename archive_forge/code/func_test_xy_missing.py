from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
@pytest.mark.parametrize('func', [funcs[0] for funcs in XY_PREDICATES])
def test_xy_missing(func):
    actual = func(np.array([point, point, point, None]), np.array([point.x, np.nan, point.x, point.x]), np.array([point.y, point.y, np.nan, point.y]))
    np.testing.assert_allclose(actual, [True, False, False, False])