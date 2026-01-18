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
def write_invalid_date_file(date_str, tmpdir, ext, engine):
    tempfilename = os.path.join(str(tmpdir), f'test_invalid_datetime.{ext}')
    df = GeoDataFrame({'date': ['2014-08-26T10:01:23', '2014-08-26T10:01:23', date_str], 'geometry': [Point(1, 1), Point(1, 1), Point(1, 1)]})
    if ext == 'geojson':
        df.to_file(tempfilename)
    else:
        schema = {'geometry': 'Point', 'properties': {'date': 'datetime'}}
        if engine == 'pyogrio' and (not fiona):
            pytest.skip('test requires fiona kwarg schema')
        df.to_file(tempfilename, schema=schema, engine='fiona')
    return tempfilename