import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_not_gdf(single_rectangle_gdf):
    """Non-GeoDataFrame inputs raise attribute errors."""
    with pytest.raises(TypeError):
        clip((2, 3), single_rectangle_gdf)
    with pytest.raises(TypeError):
        clip(single_rectangle_gdf, 'foobar')
    with pytest.raises(TypeError):
        clip(single_rectangle_gdf, (1, 2, 3))
    with pytest.raises(TypeError):
        clip(single_rectangle_gdf, (1, 2, 3, 4, 5))