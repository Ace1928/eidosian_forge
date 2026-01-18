import numpy as np
from shapely.geometry import Point
from shapely.wkt import loads
import geopandas
import pytest
from pandas.testing import assert_series_equal
def test_specified_total_bounds(geoseries_points):
    result = geoseries_points.hilbert_distance(total_bounds=geoseries_points.total_bounds)
    expected = geoseries_points.hilbert_distance()
    assert_series_equal(result, expected)