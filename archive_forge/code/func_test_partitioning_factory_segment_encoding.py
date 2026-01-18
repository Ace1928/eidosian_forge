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
@pytest.mark.parametrize('pickled', [lambda x, m: x, lambda x, m: m.loads(m.dumps(x))])
def test_partitioning_factory_segment_encoding(pickled, pickle_module):
    mockfs = fs._MockFileSystem()
    format = ds.IpcFileFormat()
    schema = pa.schema([('i64', pa.int64())])
    table = pa.table([pa.array(range(10))], schema=schema)
    partition_schema = pa.schema([('date', pa.timestamp('s')), ('string', pa.string())])
    string_partition_schema = pa.schema([('date', pa.string()), ('string', pa.string())])
    full_schema = pa.schema(list(schema) + list(partition_schema))
    for directory in ['directory/2021-05-04 00%3A00%3A00/%24', 'hive/date=2021-05-04 00%3A00%3A00/string=%24']:
        mockfs.create_dir(directory)
        with mockfs.open_output_stream(directory + '/0.feather') as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                writer.write_table(table)
                writer.close()
    selector = fs.FileSelector('directory', recursive=True)
    options = ds.FileSystemFactoryOptions('directory')
    partitioning_factory = ds.DirectoryPartitioning.discover(schema=partition_schema)
    options.partitioning_factory = pickled(partitioning_factory, pickle_module)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    inferred_schema = factory.inspect()
    assert inferred_schema == full_schema
    actual = factory.finish().to_table(columns={'date_int': ds.field('date').cast(pa.int64())})
    assert actual[0][0].as_py() == 1620086400
    partitioning_factory = ds.DirectoryPartitioning.discover(['date', 'string'], segment_encoding='none')
    options.partitioning_factory = pickled(partitioning_factory, pickle_module)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals((ds.field('date') == '2021-05-04 00%3A00%3A00') & (ds.field('string') == '%24'))
    partitioning = ds.DirectoryPartitioning(string_partition_schema, segment_encoding='none')
    options.partitioning = pickled(partitioning, pickle_module)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals((ds.field('date') == '2021-05-04 00%3A00%3A00') & (ds.field('string') == '%24'))
    partitioning_factory = ds.DirectoryPartitioning.discover(schema=partition_schema, segment_encoding='none')
    options.partitioning_factory = pickled(partitioning_factory, pickle_module)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    with pytest.raises(pa.ArrowInvalid, match='Could not cast segments for partition field'):
        inferred_schema = factory.inspect()
    selector = fs.FileSelector('hive', recursive=True)
    options = ds.FileSystemFactoryOptions('hive')
    partitioning_factory = ds.HivePartitioning.discover(schema=partition_schema)
    options.partitioning_factory = pickled(partitioning_factory, pickle_module)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    inferred_schema = factory.inspect()
    assert inferred_schema == full_schema
    actual = factory.finish().to_table(columns={'date_int': ds.field('date').cast(pa.int64())})
    assert actual[0][0].as_py() == 1620086400
    partitioning_factory = ds.HivePartitioning.discover(segment_encoding='none')
    options.partitioning_factory = pickled(partitioning_factory, pickle_module)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals((ds.field('date') == '2021-05-04 00%3A00%3A00') & (ds.field('string') == '%24'))
    options.partitioning = ds.HivePartitioning(string_partition_schema, segment_encoding='none')
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals((ds.field('date') == '2021-05-04 00%3A00%3A00') & (ds.field('string') == '%24'))
    partitioning_factory = ds.HivePartitioning.discover(schema=partition_schema, segment_encoding='none')
    options.partitioning_factory = pickled(partitioning_factory, pickle_module)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    with pytest.raises(pa.ArrowInvalid, match='Could not cast segments for partition field'):
        inferred_schema = factory.inspect()