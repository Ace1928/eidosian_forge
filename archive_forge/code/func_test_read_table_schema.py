import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
def test_read_table_schema(tempdir):
    table = pa.table({'a': pa.array([1, 2, 3], pa.int32())})
    pq.write_table(table, tempdir / 'data1.parquet')
    pq.write_table(table, tempdir / 'data2.parquet')
    schema = pa.schema([('a', 'int64')])
    result = pq.read_table(tempdir / 'data1.parquet', schema=schema)
    expected = pa.table({'a': [1, 2, 3]}, schema=schema)
    assert result.equals(expected)
    result = pq.read_table(tempdir, schema=schema)
    expected = pa.table({'a': [1, 2, 3, 1, 2, 3]}, schema=schema)
    assert result.equals(expected)
    result = pq.ParquetDataset(tempdir, schema=schema)
    expected = pa.table({'a': [1, 2, 3, 1, 2, 3]}, schema=schema)
    assert result.read().equals(expected)