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
@FIONA_MARK
def test_option_io_engine():
    try:
        geopandas.options.io_engine = 'pyogrio'
        import fiona
        orig = fiona.supported_drivers['ESRI Shapefile']
        fiona.supported_drivers['ESRI Shapefile'] = 'w'
        nybb_filename = geopandas.datasets.get_path('nybb')
        _ = geopandas.read_file(nybb_filename)
    finally:
        fiona.supported_drivers['ESRI Shapefile'] = orig
        geopandas.options.io_engine = None