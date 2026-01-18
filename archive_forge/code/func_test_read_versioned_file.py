from __future__ import absolute_import
from itertools import product
import json
from packaging.version import Version
import os
import pathlib
import pytest
from pandas import DataFrame, read_parquet as pd_read_parquet
from pandas.testing import assert_frame_equal
import numpy as np
import pyproj
import shapely
from shapely.geometry import box, Point, MultiPolygon
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, read_file, read_parquet, read_feather
from geopandas.array import to_wkb
from geopandas.datasets import get_path
from geopandas.io.arrow import (
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import mock
@pytest.mark.parametrize('version', ['0.1.0', '0.4.0', '1.0.0-beta.1'])
def test_read_versioned_file(version):
    """
    Verify that files for different metadata spec versions can be read
    created for each supported version:

    # small dummy test dataset (not naturalearth_lowres, as this can change over time)
    from shapely.geometry import box, MultiPolygon
    df = geopandas.GeoDataFrame(
        {"col_str": ["a", "b"], "col_int": [1, 2], "col_float": [0.1, 0.2]},
        geometry=[MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), box(4, 4, 5,5)],
        crs="EPSG:4326",
    )
    df.to_feather(DATA_PATH / 'arrow' / f'test_data_v{METADATA_VERSION}.feather')
    df.to_parquet(DATA_PATH / 'arrow' / f'test_data_v{METADATA_VERSION}.parquet')
    """
    expected = geopandas.GeoDataFrame({'col_str': ['a', 'b'], 'col_int': [1, 2], 'col_float': [0.1, 0.2]}, geometry=[MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), box(4, 4, 5, 5)], crs='EPSG:4326')
    df = geopandas.read_feather(DATA_PATH / 'arrow' / f'test_data_v{version}.feather')
    assert_geodataframe_equal(df, expected, check_crs=True)
    df = geopandas.read_parquet(DATA_PATH / 'arrow' / f'test_data_v{version}.parquet')
    assert_geodataframe_equal(df, expected, check_crs=True)