import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='GEOS < 3.10')
@pytest.mark.parametrize('geometry', all_types)
def test_segmentize_max_segment_length_nan(geometry):
    actual = shapely.segmentize(geometry, max_segment_length=np.nan)
    assert actual is None