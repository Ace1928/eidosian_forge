import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('geometry', all_types)
def test_snap_nan_float(geometry):
    actual = shapely.snap(geometry, point, tolerance=np.nan)
    assert actual is None