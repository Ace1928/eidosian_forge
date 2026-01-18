import numpy as np
from shapely.geometry import Point
from shapely.wkt import loads
import geopandas
import pytest
from pandas.testing import assert_series_equal
def test_zero_width():
    s = geopandas.GeoSeries([Point(0, 0), Point(0, 2), Point(0, 1)])
    with np.errstate(all='raise'):
        result = s.hilbert_distance()
    assert np.array(result).argsort().tolist() == [0, 2, 1]