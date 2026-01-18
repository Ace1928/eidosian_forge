from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.filterwarnings('ignore:The series.append method is deprecated')
@pytest.mark.skipif(compat.PANDAS_GE_20, reason='append removed in pandas 2.0')
def test_polygons_append(self):
    t1 = Polygon([(0, 0), (1, 0), (1, 1)])
    t2 = Polygon([(0, 0), (1, 1), (0, 1)])
    sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    s = GeoSeries([t1, t2, sq])
    t = GeoSeries([t1, t2, sq], [3, 4, 5])
    s = s.append(t)
    assert len(s) == 6
    assert s.sindex.size == 6