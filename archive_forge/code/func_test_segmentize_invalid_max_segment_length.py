import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='GEOS < 3.10')
@pytest.mark.parametrize('geometry', all_types)
@pytest.mark.parametrize('max_segment_length', [-1, 0])
def test_segmentize_invalid_max_segment_length(geometry, max_segment_length):
    with pytest.raises(GEOSException, match='IllegalArgumentException'):
        shapely.segmentize(geometry, max_segment_length=max_segment_length)