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
def test_preserve_attrs(df):
    df.attrs['name'] = 'my_name'
    attrs = {'name': 'my_name'}
    assert df.attrs == attrs
    for subset in [df[:2], df[df['value1'] > 2], df[['value2', 'geometry']]]:
        assert df.attrs == attrs
    df2 = df.reset_index()
    assert df2.attrs == attrs
    df3 = df2.explode(index_parts=True)
    assert df3.attrs == attrs