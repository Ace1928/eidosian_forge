from __future__ import annotations
import gzip
import os
import warnings
from io import BytesIO, StringIO
from unittest import mock
import pytest
import fsspec
from fsspec.compression import compr
from packaging.version import Version
from tlz import partition_all, valmap
import dask
from dask.base import compute_as_if_collection
from dask.bytes.core import read_bytes
from dask.bytes.utils import compress
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.io.csv import (
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import (
from dask.layers import DataFrameIOLayer
from dask.utils import filetext, filetexts, tmpdir, tmpfile
from dask.utils_test import hlg_layer
def test_reading_empty_csv_files_with_path():
    with tmpdir() as tdir:
        for k, content in enumerate(['0, 1, 2', '', '6, 7, 8']):
            with open(os.path.join(tdir, str(k) + '.csv'), 'w') as file:
                file.write(content)
        result = dd.read_csv(os.path.join(tdir, '*.csv'), include_path_column=True, converters={'path': parse_filename}, names=['A', 'B', 'C']).compute()
        df = pd.DataFrame({'A': [0, 6], 'B': [1, 7], 'C': [2, 8], 'path': ['0.csv', '2.csv']})
        df['path'] = df['path'].astype('category')
        assert_eq(result, df, check_index=False)