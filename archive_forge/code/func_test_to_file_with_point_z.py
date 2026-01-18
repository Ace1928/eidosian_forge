import datetime
import io
import os
import pathlib
import tempfile
from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
import pytz
from packaging.version import Version
from pandas.api.types import is_datetime64_any_dtype
from pandas.testing import assert_series_equal
from shapely.geometry import Point, Polygon, box
import geopandas
from geopandas import GeoDataFrame, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.io.file import _detect_driver, _EXTENSION_TO_DRIVER
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
@pytest.mark.parametrize('driver,ext', driver_ext_pairs)
def test_to_file_with_point_z(tmpdir, ext, driver, engine):
    """Test that 3D geometries are retained in writes (GH #612)."""
    tempfilename = os.path.join(str(tmpdir), 'test_3Dpoint' + ext)
    point3d = Point(0, 0, 500)
    point2d = Point(1, 1)
    df = GeoDataFrame({'a': [1, 2]}, geometry=[point3d, point2d], crs=_CRS)
    df.to_file(tempfilename, driver=driver, engine=engine)
    df_read = GeoDataFrame.from_file(tempfilename, engine=engine)
    assert_geoseries_equal(df.geometry, df_read.geometry)
    assert_correct_driver(tempfilename, ext, engine)