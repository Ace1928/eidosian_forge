import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_warning_crs_mismatch(point_gdf, single_rectangle_gdf):
    with pytest.warns(UserWarning, match='CRS mismatch between the CRS'):
        clip(point_gdf, single_rectangle_gdf.to_crs(4326))