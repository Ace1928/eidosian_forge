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
@pytest.mark.parametrize('metadata,error', [(None, 'Missing or malformed geo metadata in Parquet/Feather file'), ({}, 'Missing or malformed geo metadata in Parquet/Feather file'), ({'primary_column': 'foo', 'columns': None}, "'geo' metadata in Parquet/Feather file is missing required key"), ({'primary_column': 'foo', 'version': '<version>'}, "'geo' metadata in Parquet/Feather file is missing required key:"), ({'columns': [], 'version': '<version>'}, "'geo' metadata in Parquet/Feather file is missing required key:"), ({'primary_column': 'foo', 'columns': [], 'version': '<version>'}, "'columns' in 'geo' metadata must be a dict"), ({'primary_column': 'foo', 'columns': {'foo': {}}, 'version': '<version>'}, "'geo' metadata in Parquet/Feather file is missing required key 'encoding' for column 'foo'"), ({'primary_column': 'foo', 'columns': {'foo': {'crs': None, 'encoding': None}}, 'version': '<version>'}, 'Only WKB geometry encoding is supported'), ({'primary_column': 'foo', 'columns': {'foo': {'crs': None, 'encoding': 'BKW'}}, 'version': '<version>'}, 'Only WKB geometry encoding is supported')])
def test_validate_metadata_invalid(metadata, error):
    with pytest.raises(ValueError, match=error):
        _validate_metadata(metadata)