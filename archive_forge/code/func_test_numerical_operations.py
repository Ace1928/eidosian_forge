import os
from packaging.version import Version
import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import shapely
from shapely.geometry import Point, GeometryCollection, LineString, LinearRing
import geopandas
from geopandas import GeoDataFrame, GeoSeries
import geopandas._compat as compat
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
@pytest.mark.filterwarnings('ignore:Dropping of nuisance columns in DataFrame reductions')
def test_numerical_operations(s, df):
    exp = pd.Series([3, 4], index=['value1', 'value2'])
    if not compat.PANDAS_GE_20:
        res = df.sum()
    else:
        res = df.sum(numeric_only=True)
    assert_series_equal(res, exp)
    with pytest.raises(TypeError):
        s.sum()
    with pytest.raises(TypeError):
        s.max()
    with pytest.raises((TypeError, ValueError)):
        s.idxmax()
    with pytest.raises(TypeError):
        df + 1
    with pytest.raises(TypeError):
        s + 1
    res = df == 100
    exp = pd.DataFrame(False, index=df.index, columns=df.columns)
    assert_frame_equal(res, exp)