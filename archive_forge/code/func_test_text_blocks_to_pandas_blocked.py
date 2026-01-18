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
@csv_and_table
def test_text_blocks_to_pandas_blocked(reader, files):
    expected = read_files()
    header = files['2014-01-01.csv'].split(b'\n')[0] + b'\n'
    blocks = []
    for k in sorted(files):
        b = files[k]
        lines = b.split(b'\n')
        blocks.append([b'\n'.join(bs) for bs in partition_all(2, lines)])
    df = text_blocks_to_pandas(reader, blocks, header, expected.head(), {})
    assert_eq(df.compute().reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)
    expected2 = expected[['name', 'id']]
    df = text_blocks_to_pandas(reader, blocks, header, expected2.head(), {'usecols': ['name', 'id']})
    assert_eq(df.compute().reset_index(drop=True), expected2.reset_index(drop=True), check_dtype=False)