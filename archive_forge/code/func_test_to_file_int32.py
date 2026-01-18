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
@pytest.mark.parametrize('driver,ext', driver_ext_pairs + [('OGR_GMT', '.gmt')])
def test_to_file_int32(tmpdir, df_points, engine, driver, ext):
    tempfilename = os.path.join(str(tmpdir), f'int32.{ext}')
    geometry = df_points.geometry
    df = GeoDataFrame(geometry=geometry)
    df['data'] = pd.array([1, np.nan] * 5, dtype=pd.Int32Dtype())
    df.to_file(tempfilename, driver=driver, engine=engine)
    df_read = GeoDataFrame.from_file(tempfilename, driver=driver, engine=engine)
    assert_geodataframe_equal(df_read, df, check_dtype=False, check_like=True)
    if engine == 'pyogrio':
        tempfilename2 = os.path.join(str(tmpdir), f'int32_2.{ext}')
        df2 = df.dropna()
        df2.to_file(tempfilename2, driver=driver, engine=engine)
        df2_read = GeoDataFrame.from_file(tempfilename2, driver=driver, engine=engine)
        assert df2_read['data'].dtype == 'int32'