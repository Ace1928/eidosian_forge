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
def test_partitioning():
    schema = pa.schema([pa.field('i64', pa.int64()), pa.field('f64', pa.float64())])
    for klass in [ds.DirectoryPartitioning, ds.HivePartitioning, ds.FilenamePartitioning]:
        partitioning = klass(schema)
        assert isinstance(partitioning, ds.Partitioning)
        assert partitioning == klass(schema)
        assert partitioning != 'other object'
    schema = pa.schema([pa.field('group', pa.int64()), pa.field('key', pa.float64())])
    partitioning = ds.DirectoryPartitioning(schema)
    assert len(partitioning.dictionaries) == 2
    assert all((x is None for x in partitioning.dictionaries))
    expr = partitioning.parse('/3/3.14/')
    assert isinstance(expr, ds.Expression)
    expected = (ds.field('group') == 3) & (ds.field('key') == 3.14)
    assert expr.equals(expected)
    with pytest.raises(pa.ArrowInvalid):
        partitioning.parse('/prefix/3/aaa')
    expr = partitioning.parse('/3/')
    expected = ds.field('group') == 3
    assert expr.equals(expected)
    assert partitioning != ds.DirectoryPartitioning(schema, segment_encoding='none')
    schema = pa.schema([pa.field('alpha', pa.int64()), pa.field('beta', pa.int64())])
    partitioning = ds.HivePartitioning(schema, null_fallback='xyz')
    assert len(partitioning.dictionaries) == 2
    assert all((x is None for x in partitioning.dictionaries))
    expr = partitioning.parse('/alpha=0/beta=3/')
    expected = (ds.field('alpha') == ds.scalar(0)) & (ds.field('beta') == ds.scalar(3))
    assert expr.equals(expected)
    expr = partitioning.parse('/alpha=xyz/beta=3/')
    expected = ds.field('alpha').is_null() & (ds.field('beta') == ds.scalar(3))
    assert expr.equals(expected)
    for shouldfail in ['/alpha=one/beta=2/', '/alpha=one/', '/beta=two/']:
        with pytest.raises(pa.ArrowInvalid):
            partitioning.parse(shouldfail)
    assert partitioning != ds.HivePartitioning(schema, null_fallback='other')
    schema = pa.schema([pa.field('group', pa.int64()), pa.field('key', pa.float64())])
    partitioning = ds.FilenamePartitioning(schema)
    assert len(partitioning.dictionaries) == 2
    assert all((x is None for x in partitioning.dictionaries))
    expr = partitioning.parse('3_3.14_')
    assert isinstance(expr, ds.Expression)
    expected = (ds.field('group') == 3) & (ds.field('key') == 3.14)
    assert expr.equals(expected)
    with pytest.raises(pa.ArrowInvalid):
        partitioning.parse('prefix_3_aaa_')
    assert partitioning != ds.FilenamePartitioning(schema, segment_encoding='none')
    schema = pa.schema([pa.field('group', pa.int64()), pa.field('key', pa.dictionary(pa.int8(), pa.string()))])
    partitioning = ds.DirectoryPartitioning(schema, dictionaries={'key': pa.array(['first', 'second', 'third'])})
    assert partitioning.dictionaries[0] is None
    assert partitioning.dictionaries[1].to_pylist() == ['first', 'second', 'third']
    assert partitioning != ds.DirectoryPartitioning(schema, dictionaries=None)
    partitioning = ds.FilenamePartitioning(pa.schema([pa.field('group', pa.int64()), pa.field('key', pa.dictionary(pa.int8(), pa.string()))]), dictionaries={'key': pa.array(['first', 'second', 'third'])})
    assert partitioning.dictionaries[0] is None
    assert partitioning.dictionaries[1].to_pylist() == ['first', 'second', 'third']
    table = pa.table([pa.array(range(20)), pa.array(np.random.randn(20)), pa.array(np.repeat(['a', 'b'], 10))], names=['f1', 'f2', 'part'])
    partitioning_schema = pa.schema([('part', pa.string())])
    for klass in [ds.DirectoryPartitioning, ds.HivePartitioning, ds.FilenamePartitioning]:
        with tempfile.TemporaryDirectory() as tempdir:
            partitioning = klass(partitioning_schema)
            ds.write_dataset(table, tempdir, format='ipc', partitioning=partitioning)
            load_back = ds.dataset(tempdir, format='ipc', partitioning=partitioning)
            load_back_table = load_back.to_table()
            assert load_back_table.equals(table)