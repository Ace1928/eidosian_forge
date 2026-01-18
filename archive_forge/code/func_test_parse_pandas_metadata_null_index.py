from __future__ import annotations
import contextlib
import glob
import math
import os
import sys
import warnings
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask
import dask.dataframe as dd
import dask.multiprocessing
from dask.array.numpy_compat import NUMPY_GE_124
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import (
from dask.dataframe.io.parquet.core import get_engine
from dask.dataframe.io.parquet.utils import _parse_pandas_metadata
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
from dask.layers import DataFrameIOLayer
from dask.utils import natural_sort_key
from dask.utils_test import hlg_layer
def test_parse_pandas_metadata_null_index():
    e_index_names = [None]
    e_column_names = ['x']
    e_mapping = {'__index_level_0__': None, 'x': 'x'}
    e_column_index_names = [None]
    md = {'columns': [{'metadata': None, 'name': 'x', 'numpy_type': 'int64', 'pandas_type': 'int64'}, {'metadata': None, 'name': '__index_level_0__', 'numpy_type': 'int64', 'pandas_type': 'int64'}], 'index_columns': ['__index_level_0__'], 'pandas_version': '0.21.0'}
    index_names, column_names, mapping, column_index_names = _parse_pandas_metadata(md)
    assert index_names == e_index_names
    assert column_names == e_column_names
    assert mapping == e_mapping
    assert column_index_names == e_column_index_names
    md = {'column_indexes': [{'field_name': None, 'metadata': {'encoding': 'UTF-8'}, 'name': None, 'numpy_type': 'object', 'pandas_type': 'unicode'}], 'columns': [{'field_name': 'x', 'metadata': None, 'name': 'x', 'numpy_type': 'int64', 'pandas_type': 'int64'}, {'field_name': '__index_level_0__', 'metadata': None, 'name': None, 'numpy_type': 'int64', 'pandas_type': 'int64'}], 'index_columns': ['__index_level_0__'], 'pandas_version': '0.21.0'}
    index_names, column_names, mapping, column_index_names = _parse_pandas_metadata(md)
    assert index_names == e_index_names
    assert column_names == e_column_names
    assert mapping == e_mapping
    assert column_index_names == e_column_index_names