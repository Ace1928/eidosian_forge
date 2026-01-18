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
def test_parquet_missing_metadata2(tmpdir):
    """Missing geo metadata, such as from a parquet file created
    from a pyarrow Table (which will also not contain pandas metadata),
    will raise a ValueError.
    """
    import pyarrow.parquet as pq
    table = pyarrow.table({'a': [1, 2, 3]})
    filename = os.path.join(str(tmpdir), 'test.pq')
    pq.write_table(table, filename)
    with pytest.raises(ValueError, match='Missing geo metadata in Parquet/Feather file.'):
        read_parquet(filename)