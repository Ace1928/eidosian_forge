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
@PYOGRIO_MARK
def test_read_file__columns():
    gdf = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'), columns=['name', 'pop_est'], engine='pyogrio')
    assert gdf.columns.tolist() == ['name', 'pop_est', 'geometry']