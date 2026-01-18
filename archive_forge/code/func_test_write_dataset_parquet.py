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
def test_write_dataset_parquet(tempdir):
    table = pa.table([pa.array(range(20), type='uint32'), pa.array(np.arange('2012-01-01', 20, dtype='datetime64[D]').astype('datetime64[ns]')), pa.array(np.repeat(['a', 'b'], 10))], names=['f1', 'f2', 'part'])
    base_dir = tempdir / 'parquet_dataset'
    ds.write_dataset(table, base_dir, format='parquet')
    file_paths = list(base_dir.rglob('*'))
    expected_paths = [base_dir / 'part-0.parquet']
    assert set(file_paths) == set(expected_paths)
    result = ds.dataset(base_dir, format='parquet').to_table()
    assert result.equals(table)
    for version in ['1.0', '2.4', '2.6']:
        format = ds.ParquetFileFormat()
        opts = format.make_write_options(version=version)
        assert '<pyarrow.dataset.ParquetFileWriteOptions' in repr(opts)
        base_dir = tempdir / 'parquet_dataset_version{0}'.format(version)
        ds.write_dataset(table, base_dir, format=format, file_options=opts)
        meta = pq.read_metadata(base_dir / 'part-0.parquet')
        expected_version = '1.0' if version == '1.0' else '2.6'
        assert meta.format_version == expected_version
        result = ds.dataset(base_dir, format='parquet').to_table()
        schema = table.schema
        if version == '1.0':
            schema = schema.set(0, schema.field(0).with_type(pa.int64()))
        if version in ('1.0', '2.4'):
            schema = schema.set(1, schema.field(1).with_type(pa.timestamp('us')))
        expected = table.cast(schema)
        assert result.equals(expected)