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
@pytest.mark.parametrize('test_file', [pathlib.Path('~/test_file.geojson'), '~/test_file.geojson'])
def test_write_read_file(test_file, engine):
    gdf = geopandas.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=_CRS)
    gdf.to_file(test_file, driver='GeoJSON')
    df_json = geopandas.read_file(test_file, engine=engine)
    assert_geodataframe_equal(gdf, df_json, check_crs=True)
    os.remove(os.path.expanduser(test_file))