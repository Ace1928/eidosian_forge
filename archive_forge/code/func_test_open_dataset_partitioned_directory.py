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
@pytest.mark.parquet
def test_open_dataset_partitioned_directory(tempdir, dataset_reader, pickle_module):
    full_table, path = _create_partitioned_dataset(tempdir)
    table = full_table.select(['a', 'b'])
    _check_dataset_from_path(path, table, dataset_reader, pickle_module)
    dataset = ds.dataset(str(path), partitioning=ds.partitioning(flavor='hive'))
    assert dataset.schema.equals(full_table.schema)
    with change_cwd(tempdir):
        dataset = ds.dataset('dataset-partitioned/', partitioning=ds.partitioning(flavor='hive'))
        assert dataset.schema.equals(full_table.schema)
    dataset = ds.dataset(str(path), partitioning='hive')
    assert dataset.schema.equals(full_table.schema)
    dataset = ds.dataset(str(path), partitioning=ds.partitioning(pa.schema([('part', pa.int8())]), flavor='hive'))
    expected_schema = table.schema.append(pa.field('part', pa.int8()))
    assert dataset.schema.equals(expected_schema)
    result = dataset.to_table()
    expected = table.append_column('part', pa.array(np.repeat([0, 1, 2], 3), type=pa.int8()))
    assert result.equals(expected)