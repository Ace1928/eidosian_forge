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
@pytest.mark.parametrize('infer_dictionary', [False, True])
@pytest.mark.parametrize('pickled', [lambda x, m: x, lambda x, m: m.loads(m.dumps(x))])
def test_partitioning_factory_dictionary(mockfs, infer_dictionary, pickled, pickle_module):
    paths_or_selector = fs.FileSelector('subdir', recursive=True)
    format = ds.ParquetFileFormat()
    options = ds.FileSystemFactoryOptions('subdir')
    partitioning_factory = ds.DirectoryPartitioning.discover(['group', 'key'], infer_dictionary=infer_dictionary)
    options.partitioning_factory = pickled(partitioning_factory, pickle_module)
    factory = ds.FileSystemDatasetFactory(mockfs, paths_or_selector, format, options)
    inferred_schema = factory.inspect()
    if infer_dictionary:
        expected_type = pa.dictionary(pa.int32(), pa.string())
        assert inferred_schema.field('key').type == expected_type
        table = factory.finish().to_table().combine_chunks()
        actual = table.column('key').chunk(0)
        expected = pa.array(['xxx'] * 5 + ['yyy'] * 5).dictionary_encode()
        assert actual.equals(expected)
        table = factory.finish().to_table(filter=ds.field('key') == 'xxx')
        actual = table.column('key').chunk(0)
        expected = expected.slice(0, 5)
        assert actual.equals(expected)
    else:
        assert inferred_schema.field('key').type == pa.string()