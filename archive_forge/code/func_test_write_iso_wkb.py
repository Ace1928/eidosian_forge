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
def test_write_iso_wkb(tmpdir):
    gdf = geopandas.GeoDataFrame(geometry=geopandas.GeoSeries.from_wkt(['POINT Z (1 2 3)']))
    if compat.USE_SHAPELY_20 and shapely.geos.geos_version > (3, 10, 0):
        gdf.to_parquet(tmpdir / 'test.parquet')
    else:
        with pytest.warns(UserWarning, match='The GeoDataFrame contains 3D geometries'):
            gdf.to_parquet(tmpdir / 'test.parquet')
    from pyarrow.parquet import read_table
    table = read_table(tmpdir / 'test.parquet')
    wkb = table['geometry'][0].as_py().hex()
    if compat.USE_SHAPELY_20 and shapely.geos.geos_version > (3, 10, 0):
        assert wkb == '01e9030000000000000000f03f00000000000000400000000000000840'
    else:
        assert wkb == '0101000080000000000000f03f00000000000000400000000000000840'