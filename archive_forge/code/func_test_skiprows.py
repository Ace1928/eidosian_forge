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
@pytest.mark.parametrize('dd_read,pd_read,files', [(dd.read_csv, pd.read_csv, csv_files), (dd.read_table, pd.read_table, tsv_files)])
def test_skiprows(dd_read, pd_read, files):
    files = {name: comment_header + b'\n' + content for name, content in files.items()}
    skip = len(comment_header.splitlines())
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv', skiprows=skip)
        expected_df = read_files_with(files, pd_read, skiprows=skip)
        assert_eq(df, expected_df, check_dtype=False)