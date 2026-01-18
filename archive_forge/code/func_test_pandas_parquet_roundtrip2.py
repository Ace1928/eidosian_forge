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
@pytest.mark.parametrize('test_dataset', ['naturalearth_lowres', 'naturalearth_cities', 'nybb'])
def test_pandas_parquet_roundtrip2(test_dataset, tmpdir):
    test_dataset = 'naturalearth_lowres'
    df = DataFrame(read_file(get_path(test_dataset)).drop(columns=['geometry']))
    filename = os.path.join(str(tmpdir), 'test.pq')
    df.to_parquet(filename)
    pq_df = pd_read_parquet(filename)
    assert_frame_equal(df, pq_df)