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
def test_parse_pandas_metadata(pandas_metadata):
    index_names, column_names, mapping, column_index_names = _parse_pandas_metadata(pandas_metadata)
    assert index_names == ['idx']
    assert column_names == ['A']
    assert column_index_names == [None]
    if pandas_metadata['index_columns'] == ['__index_level_0__']:
        assert mapping == {'__index_level_0__': 'idx', 'A': 'A'}
    else:
        assert mapping == {'idx': 'idx', 'A': 'A'}
    assert isinstance(mapping, dict)