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
def test_write_dataset_preserve_field_metadata(tempdir):
    schema_metadata = pa.schema([pa.field('x', pa.int64(), metadata={b'foo': b'bar'}), pa.field('y', pa.int64())])
    schema_no_meta = pa.schema([pa.field('x', pa.int64()), pa.field('y', pa.int64())])
    arrays = [[1, 2, 3], [None, 5, None]]
    table = pa.Table.from_arrays(arrays, schema=schema_metadata)
    table_no_meta = pa.Table.from_arrays(arrays, schema=schema_no_meta)
    ds.write_dataset([table, table_no_meta], tempdir / 'test1', format='parquet')
    dataset = ds.dataset(tempdir / 'test1', format='parquet')
    assert dataset.to_table().schema.equals(schema_metadata, check_metadata=True)
    ds.write_dataset([table_no_meta, table], tempdir / 'test2', format='parquet')
    dataset = ds.dataset(tempdir / 'test2', format='parquet')
    assert dataset.to_table().schema.equals(schema_no_meta, check_metadata=True)
    ds.write_dataset([table_no_meta, table], tempdir / 'test3', format='parquet', schema=schema_metadata)
    dataset = ds.dataset(tempdir / 'test3', format='parquet')
    assert dataset.to_table().schema.equals(schema_metadata, check_metadata=True)