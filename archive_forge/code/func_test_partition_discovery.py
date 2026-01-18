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
@pytest.mark.parametrize('partitioning', ['directory', 'hive'])
@pytest.mark.parametrize('null_fallback', ['xyz', None])
@pytest.mark.parametrize('infer_dictionary', [False, True])
@pytest.mark.parametrize('partition_keys', [(['A', 'B', 'C'], [1, 2, 3]), ([1, 2, 3], ['A', 'B', 'C']), (['A', 'B', 'C'], ['D', 'E', 'F']), ([1, 2, 3], [4, 5, 6]), ([1, None, 3], ['A', 'B', 'C']), ([1, 2, 3], ['A', None, 'C']), ([None, 2, 3], [None, 2, 3])])
def test_partition_discovery(tempdir, partitioning, null_fallback, infer_dictionary, partition_keys):
    table = pa.table({'a': range(9), 'b': [0.0] * 4 + [1.0] * 5})
    has_null = None in partition_keys[0] or None in partition_keys[1]
    if partitioning == 'directory' and has_null:
        return
    if partitioning == 'directory':
        partitioning = ds.DirectoryPartitioning.discover(['part1', 'part2'], infer_dictionary=infer_dictionary)
        fmt = '{0}/{1}'
        null_value = None
    else:
        if null_fallback:
            partitioning = ds.HivePartitioning.discover(infer_dictionary=infer_dictionary, null_fallback=null_fallback)
        else:
            partitioning = ds.HivePartitioning.discover(infer_dictionary=infer_dictionary)
        fmt = 'part1={0}/part2={1}'
        if null_fallback:
            null_value = null_fallback
        else:
            null_value = '__HIVE_DEFAULT_PARTITION__'
    basepath = tempdir / 'dataset'
    basepath.mkdir()
    part_keys1, part_keys2 = partition_keys
    for part1 in part_keys1:
        for part2 in part_keys2:
            path = basepath / fmt.format(part1 or null_value, part2 or null_value)
            path.mkdir(parents=True)
            pq.write_table(table, path / 'test.parquet')
    dataset = ds.dataset(str(basepath), partitioning=partitioning)

    def expected_type(key):
        if infer_dictionary:
            value_type = pa.string() if isinstance(key, str) else pa.int32()
            return pa.dictionary(pa.int32(), value_type)
        else:
            return pa.string() if isinstance(key, str) else pa.int32()
    expected_schema = table.schema.append(pa.field('part1', expected_type(part_keys1[0]))).append(pa.field('part2', expected_type(part_keys2[0])))
    assert dataset.schema.equals(expected_schema)