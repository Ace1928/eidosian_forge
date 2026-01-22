import warnings
import pandas as pd
import pyproj
import pytest
from geopandas._compat import PANDAS_GE_21
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_index_equal
from shapely.geometry import Point
from geopandas import GeoDataFrame, GeoSeries
def _check_metadata(self, gdf, geometry_column_name='geometry', crs=None):
    assert gdf._geometry_column_name == geometry_column_name
    assert gdf.crs == crs