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
def test_to_file_crs(tmpdir, engine):
    """
    Ensure that the file is written according to the crs
    if it is specified
    """
    df = read_file(geopandas.datasets.get_path('nybb'), engine=engine)
    tempfilename = os.path.join(str(tmpdir), 'crs.shp')
    df.to_file(tempfilename, engine=engine)
    result = GeoDataFrame.from_file(tempfilename, engine=engine)
    assert result.crs == df.crs
    if engine == 'pyogrio':
        with pytest.raises(ValueError, match="Passing 'crs' it not supported"):
            df.to_file(tempfilename, crs=3857, engine=engine)
        return
    df.to_file(tempfilename, crs=3857, engine=engine)
    result = GeoDataFrame.from_file(tempfilename, engine=engine)
    assert result.crs == 'epsg:3857'
    df2 = df.copy()
    df2.crs = None
    df2.to_file(tempfilename, crs=2263, engine=engine)
    df = GeoDataFrame.from_file(tempfilename, engine=engine)
    assert df.crs == 'epsg:2263'