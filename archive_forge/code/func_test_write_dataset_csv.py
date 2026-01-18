import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
def test_write_dataset_csv(tempdir):
    table = pa.table([pa.array(range(20)), pa.array(np.random.randn(20)), pa.array(np.repeat(['a', 'b'], 10))], names=['f1', 'f2', 'chr1'])
    base_dir = tempdir / 'csv_dataset'
    ds.write_dataset(table, base_dir, format='csv')
    file_paths = list(base_dir.rglob('*'))
    expected_paths = [base_dir / 'part-0.csv']
    assert set(file_paths) == set(expected_paths)
    result = ds.dataset(base_dir, format='csv').to_table()
    assert result.equals(table)
    format = ds.CsvFileFormat(read_options=pyarrow.csv.ReadOptions(column_names=table.schema.names))
    opts = format.make_write_options(include_header=False)
    base_dir = tempdir / 'csv_dataset_noheader'
    ds.write_dataset(table, base_dir, format=format, file_options=opts)
    result = ds.dataset(base_dir, format=format).to_table()
    assert result.equals(table)