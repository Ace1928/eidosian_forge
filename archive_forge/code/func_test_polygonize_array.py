import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_polygonize_array():
    lines = [LineString([(0, 0), (1, 1)]), LineString([(0, 0), (0, 1)]), LineString([(0, 1), (1, 1)])]
    expected = GeometryCollection([Polygon([(1, 1), (0, 0), (0, 1), (1, 1)])])
    result = shapely.polygonize(np.array(lines))
    assert isinstance(result, shapely.Geometry)
    assert result == expected
    result = shapely.polygonize(np.array([lines]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result[0] == expected
    arr = np.array([lines, lines])
    assert arr.shape == (2, 3)
    result = shapely.polygonize(arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert result[0] == expected
    assert result[1] == expected
    arr = np.array([[lines, lines], [lines, lines], [lines, lines]])
    assert arr.shape == (3, 2, 3)
    result = shapely.polygonize(arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)
    for res in result.flatten():
        assert res == expected