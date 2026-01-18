import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
def test_to_crs_dimension_z():
    arr = points_from_xy([1, 2], [2, 3], [3, 4], crs=4326)
    assert arr.has_z.all()
    result = arr.to_crs(epsg=3857)
    assert result.has_z.all()