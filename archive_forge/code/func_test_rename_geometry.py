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
def test_rename_geometry(self):
    assert self.df.geometry.name == 'geometry'
    df2 = self.df.rename_geometry('new_name')
    assert df2.geometry.name == 'new_name'
    df2 = self.df.rename_geometry('new_name', inplace=True)
    assert df2 is None
    assert self.df.geometry.name == 'new_name'
    msg = 'Column named Shape_Area already exists'
    with pytest.raises(ValueError, match=msg):
        df2 = self.df.rename_geometry('Shape_Area')
    with pytest.raises(ValueError, match=msg):
        self.df.rename_geometry('Shape_Area', inplace=True)