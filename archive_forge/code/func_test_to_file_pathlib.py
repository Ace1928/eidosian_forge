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
def test_to_file_pathlib(tmpdir, df_nybb, driver, ext, engine):
    """Test to_file and from_file"""
    temppath = pathlib.Path(os.path.join(str(tmpdir), 'boros.' + ext))
    df_nybb.to_file(temppath, driver=driver, engine=engine)
    df = GeoDataFrame.from_file(temppath, engine=engine)
    assert 'geometry' in df
    assert len(df) == 5
    assert np.alltrue(df['BoroName'].values == df_nybb['BoroName'])
    assert_correct_driver(temppath, ext, engine)