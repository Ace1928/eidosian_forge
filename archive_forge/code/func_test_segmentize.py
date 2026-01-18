import platform
import weakref
import numpy as np
import pytest
import shapely
from shapely import (
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal
@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='GEOS < 3.10')
def test_segmentize():
    line = LineString([(0, 0), (0, 10)])
    result = line.segmentize(max_segment_length=5)
    assert result.equals(LineString([(0, 0), (0, 5), (0, 10)]))