from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
@pytest.mark.parametrize('func', UNARY_PREDICATES)
def test_unary_with_kwargs(func):
    out = np.empty((), dtype=np.uint8)
    actual = func(point, out=out)
    assert actual is out
    assert actual.dtype == np.uint8