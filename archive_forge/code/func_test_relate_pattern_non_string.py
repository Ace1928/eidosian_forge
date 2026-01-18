from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
@pytest.mark.parametrize('pattern', [b'*********', 10, None])
def test_relate_pattern_non_string(pattern):
    with pytest.raises(TypeError, match='expected string'):
        shapely.relate_pattern(point, polygon, pattern)