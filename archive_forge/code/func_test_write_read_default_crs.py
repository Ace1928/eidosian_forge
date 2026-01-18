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
@pytest.mark.parametrize('format', ['feather', 'parquet'])
def test_write_read_default_crs(tmpdir, format):
    if format == 'feather':
        from pyarrow.feather import write_feather as write
    else:
        from pyarrow.parquet import write_table as write
    filename = os.path.join(str(tmpdir), f'test.{format}')
    gdf = geopandas.GeoDataFrame(geometry=[box(0, 0, 10, 10)])
    table = _geopandas_to_arrow(gdf)
    metadata = table.schema.metadata
    geo_metadata = _decode_metadata(metadata[b'geo'])
    del geo_metadata['columns']['geometry']['crs']
    metadata.update({b'geo': _encode_metadata(geo_metadata)})
    table = table.replace_schema_metadata(metadata)
    write(table, filename)
    read = getattr(geopandas, f'read_{format}')
    df = read(filename)
    assert df.crs.equals(pyproj.CRS('OGC:CRS84'))