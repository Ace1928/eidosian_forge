import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(np.__version__ < '1.15', reason='axis keyword for generalized ufunc introduced in np 1.15')
def test_polygonize_array_axis():
    lines = [LineString([(0, 0), (1, 1)]), LineString([(0, 0), (0, 1)]), LineString([(0, 1), (1, 1)])]
    arr = np.array([lines, lines])
    result = shapely.polygonize(arr, axis=1)
    assert result.shape == (2,)
    result = shapely.polygonize(arr, axis=0)
    assert result.shape == (3,)