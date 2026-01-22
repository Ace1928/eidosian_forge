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
def _create_dataset_for_fragments(tempdir, chunk_size=None, filesystem=None):
    table = pa.table([range(8), [1] * 8, ['a'] * 4 + ['b'] * 4], names=['f1', 'f2', 'part'])
    path = str(tempdir / 'test_parquet_dataset')
    pq.write_to_dataset(table, path, partition_cols=['part'], chunk_size=chunk_size)
    dataset = ds.dataset(path, format='parquet', partitioning='hive', filesystem=filesystem)
    return (table, dataset)