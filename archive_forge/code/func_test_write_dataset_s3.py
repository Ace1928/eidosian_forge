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
@pytest.mark.s3
def test_write_dataset_s3(s3_example_simple):
    _, _, fs, _, host, port, access_key, secret_key = s3_example_simple
    uri_template = 's3://{}:{}@{{}}?scheme=http&endpoint_override={}:{}'.format(access_key, secret_key, host, port)
    table = pa.table([pa.array(range(20)), pa.array(np.random.randn(20)), pa.array(np.repeat(['a', 'b'], 10))], names=['f1', 'f2', 'part'])
    part = ds.partitioning(pa.schema([('part', pa.string())]), flavor='hive')
    ds.write_dataset(table, 'mybucket/dataset', filesystem=fs, format='feather', partitioning=part)
    result = ds.dataset('mybucket/dataset', filesystem=fs, format='ipc', partitioning='hive').to_table()
    assert result.equals(table)
    uri = uri_template.format('mybucket/dataset2')
    ds.write_dataset(table, uri, format='feather', partitioning=part)
    result = ds.dataset('mybucket/dataset2', filesystem=fs, format='ipc', partitioning='hive').to_table()
    assert result.equals(table)
    uri = uri_template.format('mybucket')
    ds.write_dataset(table, 'dataset3', filesystem=uri, format='feather', partitioning=part)
    result = ds.dataset('mybucket/dataset3', filesystem=fs, format='ipc', partitioning='hive').to_table()
    assert result.equals(table)