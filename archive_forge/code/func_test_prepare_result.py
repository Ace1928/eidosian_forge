import pandas as pd
from shapely.geometry import Point
from geopandas import GeoDataFrame, GeoSeries
from geopandas.tools import geocode, reverse_geocode
from geopandas.tools.geocoding import _prepare_geocode_result
from geopandas.tests.util import assert_geoseries_equal, mock
from pandas.testing import assert_series_equal
from geopandas.testing import assert_geodataframe_equal
import pytest
def test_prepare_result():
    p0 = Point(12.3, -45.6)
    p1 = Point(-23.4, 56.7)
    d = {'a': ('address0', p0.coords[0]), 'b': ('address1', p1.coords[0])}
    df = _prepare_geocode_result(d)
    assert type(df) is GeoDataFrame
    assert df.crs == 'EPSG:4326'
    assert len(df) == 2
    assert 'address' in df
    coords = df.loc['a']['geometry'].coords[0]
    test = p0.coords[0]
    assert coords[0] == pytest.approx(test[1])
    assert coords[1] == pytest.approx(test[0])
    coords = df.loc['b']['geometry'].coords[0]
    test = p1.coords[0]
    assert coords[0] == pytest.approx(test[1])
    assert coords[1] == pytest.approx(test[0])