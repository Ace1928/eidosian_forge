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
def test_parquet_read_partitioned_dataset_fsspec(tmpdir):
    fsspec = pytest.importorskip('fsspec')
    df = read_file(get_path('naturalearth_lowres'))
    memfs = fsspec.filesystem('memory')
    memfs.mkdir('partitioned_dataset')
    with memfs.open('partitioned_dataset/data1.parquet', 'wb') as f:
        df[:100].to_parquet(f)
    with memfs.open('partitioned_dataset/data2.parquet', 'wb') as f:
        df[100:].to_parquet(f)
    result = read_parquet('memory://partitioned_dataset')
    assert_geodataframe_equal(result, df)