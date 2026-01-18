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
def test_write_table_multiple_fragments(tempdir):
    table = pa.table([pa.array(range(10)), pa.array(np.random.randn(10)), pa.array(np.repeat(['a', 'b'], 5))], names=['f1', 'f2', 'part'])
    table = pa.concat_tables([table] * 2)
    base_dir = tempdir / 'single'
    ds.write_dataset(table, base_dir, format='feather')
    assert set(base_dir.rglob('*')) == set([base_dir / 'part-0.feather'])
    assert ds.dataset(base_dir, format='ipc').to_table().equals(table)
    base_dir = tempdir / 'single-list'
    ds.write_dataset([table], base_dir, format='feather')
    assert set(base_dir.rglob('*')) == set([base_dir / 'part-0.feather'])
    assert ds.dataset(base_dir, format='ipc').to_table().equals(table)
    base_dir = tempdir / 'multiple'
    ds.write_dataset(table.to_batches(), base_dir, format='feather')
    assert set(base_dir.rglob('*')) == set([base_dir / 'part-0.feather'])
    assert ds.dataset(base_dir, format='ipc').to_table().equals(table)
    base_dir = tempdir / 'multiple-table'
    ds.write_dataset([table, table], base_dir, format='feather')
    assert set(base_dir.rglob('*')) == set([base_dir / 'part-0.feather'])
    assert ds.dataset(base_dir, format='ipc').to_table().equals(pa.concat_tables([table] * 2))