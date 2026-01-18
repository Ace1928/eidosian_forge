import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import Point, Polygon
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, points_from_xy, read_file
from geopandas.array import GeometryArray, GeometryDtype, from_shapely
from geopandas._compat import ignore_shapely2_warnings
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytest
def test_set_geometry_col(self):
    g = self.df.geometry
    g_simplified = g.simplify(100)
    self.df['simplified_geometry'] = g_simplified
    df2 = self.df.set_geometry('simplified_geometry')
    assert 'simplified_geometry' in df2
    assert_geoseries_equal(df2.geometry, g_simplified)
    df3 = self.df.set_geometry('simplified_geometry', drop=True)
    assert 'simplified_geometry' not in df3
    assert_geoseries_equal(df3.geometry, g_simplified)