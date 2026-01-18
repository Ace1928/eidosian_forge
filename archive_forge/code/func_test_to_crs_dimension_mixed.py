import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
def test_to_crs_dimension_mixed():
    s = GeoSeries([Point(1, 2), LineString([(1, 2, 3), (4, 5, 6)])], crs=2056)
    result = s.to_crs(epsg=4326)
    assert not result[0].is_empty
    assert result.has_z.tolist() == [False, True]
    roundtrip = result.to_crs(epsg=2056)
    for a, b in zip(roundtrip, s):
        np.testing.assert_allclose(a.coords[:], b.coords[:], atol=0.01)