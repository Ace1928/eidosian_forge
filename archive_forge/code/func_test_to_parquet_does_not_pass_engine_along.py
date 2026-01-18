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
@mock.patch('geopandas.io.arrow._to_parquet')
def test_to_parquet_does_not_pass_engine_along(mock_to_parquet):
    df = GeoDataFrame(data=[[1, 2, 3]], columns=['a', 'b', 'a'], geometry=[Point(1, 1)])
    df.to_parquet('', engine='pyarrow')
    mock_to_parquet.assert_called_with(df, '', compression='snappy', index=None, schema_version=None)