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
def test_specified_schema(tempdir, dataset_reader):
    table = pa.table({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]})
    pq.write_table(table, tempdir / 'data.parquet')

    def _check_dataset(schema, expected, expected_schema=None):
        dataset = ds.dataset(str(tempdir / 'data.parquet'), schema=schema)
        if expected_schema is not None:
            assert dataset.schema.equals(expected_schema)
        else:
            assert dataset.schema.equals(schema)
        result = dataset_reader.to_table(dataset)
        assert result.equals(expected)
    schema = None
    expected = table
    _check_dataset(schema, expected, expected_schema=table.schema)
    schema = table.schema
    expected = table
    _check_dataset(schema, expected)
    schema = pa.schema([('b', 'float64'), ('a', 'int64')])
    expected = pa.table([[0.1, 0.2, 0.3], [1, 2, 3]], names=['b', 'a'])
    _check_dataset(schema, expected)
    schema = pa.schema([('a', 'int64')])
    expected = pa.table([[1, 2, 3]], names=['a'])
    _check_dataset(schema, expected)
    schema = pa.schema([('a', 'int64'), ('c', 'int32')])
    expected = pa.table([[1, 2, 3], pa.array([None, None, None], type='int32')], names=['a', 'c'])
    _check_dataset(schema, expected)
    schema = pa.schema([('a', 'int32'), ('b', 'float64')])
    dataset = ds.dataset(str(tempdir / 'data.parquet'), schema=schema)
    expected = pa.table([table['a'].cast('int32'), table['b']], names=['a', 'b'])
    _check_dataset(schema, expected)
    schema = pa.schema([('a', pa.list_(pa.int32())), ('b', 'float64')])
    dataset = ds.dataset(str(tempdir / 'data.parquet'), schema=schema)
    assert dataset.schema.equals(schema)
    with pytest.raises(NotImplementedError, match='Unsupported cast from int64 to list'):
        dataset_reader.to_table(dataset)