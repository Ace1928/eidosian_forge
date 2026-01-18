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
@pytest.mark.parametrize('format,version', product(['feather', 'parquet'], [None] + SUPPORTED_VERSIONS))
def test_write_deprecated_version_parameter(tmpdir, format, version):
    if format == 'feather':
        from pyarrow.feather import read_table
        version = version or 2
    else:
        from pyarrow.parquet import read_table
        version = version or '2.6'
    filename = os.path.join(str(tmpdir), f'test.{format}')
    gdf = geopandas.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs='EPSG:4326')
    write = getattr(gdf, f'to_{format}')
    if version in SUPPORTED_VERSIONS:
        with pytest.warns(FutureWarning, match='the `version` parameter has been replaced with `schema_version`'):
            write(filename, version=version)
    else:
        write(filename, version=version)
    table = read_table(filename)
    metadata = json.loads(table.schema.metadata[b'geo'])
    if version in SUPPORTED_VERSIONS:
        assert metadata['version'] == version
    else:
        assert metadata['version'] == METADATA_VERSION