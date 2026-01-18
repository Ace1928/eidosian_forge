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
def test_write_scanner(tempdir, dataset_reader):
    table = pa.table([pa.array(range(20)), pa.array(np.random.randn(20)), pa.array(np.repeat(['a', 'b'], 10))], names=['f1', 'f2', 'part'])
    dataset = ds.dataset(table)
    base_dir = tempdir / 'dataset_from_scanner'
    ds.write_dataset(dataset_reader.scanner(dataset), base_dir, format='feather')
    result = dataset_reader.to_table(ds.dataset(base_dir, format='ipc'))
    assert result.equals(table)
    base_dir = tempdir / 'dataset_from_scanner2'
    ds.write_dataset(dataset_reader.scanner(dataset, columns=['f1']), base_dir, format='feather')
    result = dataset_reader.to_table(ds.dataset(base_dir, format='ipc'))
    assert result.equals(table.select(['f1']))
    with pytest.raises(ValueError, match='Cannot specify a schema'):
        ds.write_dataset(dataset_reader.scanner(dataset), base_dir, schema=table.schema, format='feather')