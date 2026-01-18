import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
@pytest.mark.parametrize('constructor', [lambda geoms, crs: GeoSeries(geoms, crs=crs), lambda geoms, crs: GeoDataFrame(geometry=geoms, crs=crs)], ids=['geoseries', 'geodataframe'])
def test_set_crs(self, constructor):
    naive = constructor([Point(0, 0), Point(1, 1)], crs=None)
    assert naive.crs is None
    result = naive.set_crs(crs='EPSG:4326')
    assert result.crs == 'EPSG:4326'
    assert naive.crs is None
    result = naive.set_crs(epsg=4326)
    assert result.crs == 'EPSG:4326'
    assert naive.crs is None
    result = naive.set_crs(crs='EPSG:4326', inplace=True)
    assert result is naive
    assert result.crs == naive.crs == 'EPSG:4326'
    non_naive = constructor([Point(0, 0), Point(1, 1)], crs='EPSG:4326')
    assert non_naive.crs == 'EPSG:4326'
    with pytest.raises(ValueError, match='already has a CRS'):
        non_naive.set_crs('EPSG:3857')
    result = non_naive.set_crs('EPSG:4326')
    assert result.crs == 'EPSG:4326'
    result = non_naive.set_crs('EPSG:3857', allow_override=True)
    assert non_naive.crs == 'EPSG:4326'
    assert result.crs == 'EPSG:3857'
    result = non_naive.set_crs('EPSG:3857', allow_override=True, inplace=True)
    assert non_naive.crs == 'EPSG:3857'
    assert result.crs == 'EPSG:3857'
    with pytest.raises(ValueError):
        naive.set_crs(crs=None, epsg=None)