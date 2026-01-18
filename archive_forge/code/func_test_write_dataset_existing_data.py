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
@pytest.mark.pandas
def test_write_dataset_existing_data(tempdir):
    directory = tempdir / 'ds'
    table = pa.table({'b': ['x', 'y', 'z'], 'c': [1, 2, 3]})
    partitioning = ds.partitioning(schema=pa.schema([pa.field('c', pa.int64())]), flavor='hive')

    def compare_tables_ignoring_order(t1, t2):
        df1 = t1.to_pandas().sort_values('b').reset_index(drop=True)
        df2 = t2.to_pandas().sort_values('b').reset_index(drop=True)
        assert df1.equals(df2)
    ds.write_dataset(table, directory, partitioning=partitioning, format='ipc')
    table = pa.table({'b': ['a', 'b', 'c'], 'c': [2, 3, 4]})
    with pytest.raises(pa.ArrowInvalid):
        ds.write_dataset(table, directory, partitioning=partitioning, format='ipc')
    extra_table = pa.table({'b': ['e']})
    extra_file = directory / 'c=2' / 'foo.arrow'
    pyarrow.feather.write_feather(extra_table, extra_file)
    ds.write_dataset(table, directory, partitioning=partitioning, format='ipc', existing_data_behavior='overwrite_or_ignore')
    overwritten = pa.table({'b': ['e', 'x', 'a', 'b', 'c'], 'c': [2, 1, 2, 3, 4]})
    readback = ds.dataset(tempdir, format='ipc', partitioning=partitioning).to_table()
    compare_tables_ignoring_order(readback, overwritten)
    assert extra_file.exists()
    ds.write_dataset(table, directory, partitioning=partitioning, format='ipc', existing_data_behavior='delete_matching')
    overwritten = pa.table({'b': ['x', 'a', 'b', 'c'], 'c': [1, 2, 3, 4]})
    readback = ds.dataset(tempdir, format='ipc', partitioning=partitioning).to_table()
    compare_tables_ignoring_order(readback, overwritten)
    assert not extra_file.exists()