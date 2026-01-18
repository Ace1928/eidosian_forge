import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(np.__version__ < '1.15', reason='axis keyword for generalized ufunc introduced in np 1.15')
def test_polygonize_full_array_axis():
    lines = [LineString([(0, 0), (1, 1)]), LineString([(0, 0), (0, 1)]), LineString([(0, 1), (1, 1)])]
    arr = np.array([lines, lines])
    result = shapely.polygonize_full(arr, axis=1)
    assert len(result) == 4
    assert all((arr.shape == (2,) for arr in result))
    result = shapely.polygonize_full(arr, axis=0)
    assert len(result) == 4
    assert all((arr.shape == (3,) for arr in result))