import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_warning_extra_geoms_mixed(self, mixed_gdf, mask):
    """Test the correct warnings are raised if keep_geom_type is
        called on a mixed GDF"""
    with pytest.warns(UserWarning):
        clip(mixed_gdf, mask, keep_geom_type=True)