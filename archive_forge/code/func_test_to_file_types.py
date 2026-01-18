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
def test_to_file_types(tmpdir, df_points, engine):
    """Test various integer type columns (GH#93)"""
    tempfilename = os.path.join(str(tmpdir), 'int.shp')
    int_types = [np.int8, np.int16, np.int32, np.int64, np.intp, np.uint8, np.uint16, np.uint32, np.uint64]
    geometry = df_points.geometry
    data = {str(i): np.arange(len(geometry), dtype=dtype) for i, dtype in enumerate(int_types)}
    df = GeoDataFrame(data, geometry=geometry)
    df.to_file(tempfilename, engine=engine)