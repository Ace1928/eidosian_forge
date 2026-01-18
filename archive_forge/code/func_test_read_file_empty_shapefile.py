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
@pytest.mark.filterwarnings("ignore:Layer 'b'test_empty'' does not have any features:UserWarning")
def test_read_file_empty_shapefile(tmpdir, engine):
    if engine == 'pyogrio' and (not fiona):
        pytest.skip('test requires fiona to work')
    from geopandas.io.file import fiona_env
    meta = {'crs': {}, 'crs_wkt': '', 'driver': 'ESRI Shapefile', 'schema': {'geometry': 'Point', 'properties': OrderedDict([('A', 'int:9'), ('Z', 'float:24.15')])}}
    fname = str(tmpdir.join('test_empty.shp'))
    with fiona_env():
        with fiona.open(fname, 'w', **meta) as _:
            pass
    empty = read_file(fname, engine=engine)
    assert isinstance(empty, geopandas.GeoDataFrame)
    assert all(empty.columns == ['A', 'Z', 'geometry'])