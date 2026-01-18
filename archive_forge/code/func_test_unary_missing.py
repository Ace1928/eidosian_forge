from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
@pytest.mark.parametrize('func', UNARY_PREDICATES)
def test_unary_missing(func):
    if func in (shapely.is_valid_input, shapely.is_missing):
        assert func(None)
    else:
        assert not func(None)