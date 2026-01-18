import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_returns_series(self, point_gdf, mask):
    """Test that function returns a GeoSeries if GeoSeries is passed."""
    out = clip(point_gdf.geometry, mask)
    assert isinstance(out, GeoSeries)