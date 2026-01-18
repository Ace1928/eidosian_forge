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
def test_to_json_dropna(self):
    self.df.loc[self.df['BoroName'] == 'Queens', 'Shape_Area'] = np.nan
    self.df.loc[self.df['BoroName'] == 'Bronx', 'Shape_Leng'] = np.nan
    text = self.df.to_json(na='drop')
    data = json.loads(text)
    assert len(data['features']) == 5
    for f in data['features']:
        props = f['properties']
        if props['BoroName'] == 'Queens':
            assert len(props) == 3
            assert 'Shape_Area' not in props
            assert 'Shape_Leng' in props
        elif props['BoroName'] == 'Bronx':
            assert len(props) == 3
            assert 'Shape_Leng' not in props
            assert 'Shape_Area' in props
        else:
            assert len(props) == 4