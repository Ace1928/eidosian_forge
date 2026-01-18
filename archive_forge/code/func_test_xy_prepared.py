from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
@pytest.mark.parametrize('func, func_bin', XY_PREDICATES)
def test_xy_prepared(func, func_bin):
    actual = func(_prepare_with_copy([polygon, line_string]), 2, 3)
    expected = func_bin([polygon, line_string], Point(2, 3))
    np.testing.assert_allclose(actual, expected)