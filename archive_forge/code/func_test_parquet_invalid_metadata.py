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
@pytest.mark.parametrize('geo_meta,error', [({'geo': b''}, 'Missing or malformed geo metadata in Parquet/Feather file'), ({'geo': _encode_metadata({})}, 'Missing or malformed geo metadata in Parquet/Feather file'), ({'geo': _encode_metadata({'foo': 'bar'})}, "'geo' metadata in Parquet/Feather file is missing required key")])
def test_parquet_invalid_metadata(tmpdir, geo_meta, error):
    """Has geo metadata with missing required fields will raise a ValueError.

    This requires writing the parquet file directly below, so that we can
    control the metadata that is written for this test.
    """
    from pyarrow import parquet, Table
    test_dataset = 'naturalearth_lowres'
    df = read_file(get_path(test_dataset))
    df = DataFrame(df)
    df['geometry'] = to_wkb(df['geometry'].values)
    table = Table.from_pandas(df)
    metadata = table.schema.metadata
    metadata.update(geo_meta)
    table = table.replace_schema_metadata(metadata)
    filename = os.path.join(str(tmpdir), 'test.pq')
    parquet.write_table(table, filename)
    with pytest.raises(ValueError, match=error):
        read_parquet(filename)