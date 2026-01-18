import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='GEOS < 3.10')
def test_segmentize_none():
    assert shapely.segmentize(None, max_segment_length=5) is None