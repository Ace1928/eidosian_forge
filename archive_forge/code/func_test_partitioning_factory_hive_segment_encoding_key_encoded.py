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
def test_partitioning_factory_hive_segment_encoding_key_encoded(pickled, pickle_module):
    mockfs = fs._MockFileSystem()
    format = ds.IpcFileFormat()
    schema = pa.schema([('i64', pa.int64())])
    table = pa.table([pa.array(range(10))], schema=schema)
    partition_schema = pa.schema([("test'; date", pa.timestamp('s')), ("test';[ string'", pa.string())])
    string_partition_schema = pa.schema([("test'; date", pa.string()), ("test';[ string'", pa.string())])
    full_schema = pa.schema(list(schema) + list(partition_schema))
    partition_schema_en = pa.schema([('test%27%3B%20date', pa.timestamp('s')), ('test%27%3B%5B%20string%27', pa.string())])
    string_partition_schema_en = pa.schema([('test%27%3B%20date', pa.string()), ('test%27%3B%5B%20string%27', pa.string())])
    directory = 'hive/test%27%3B%20date=2021-05-04 00%3A00%3A00/test%27%3B%5B%20string%27=%24'
    mockfs.create_dir(directory)
    with mockfs.open_output_stream(directory + '/0.feather') as sink:
        with pa.ipc.new_file(sink, schema) as writer:
            writer.write_table(table)
            writer.close()
    selector = fs.FileSelector('hive', recursive=True)
    options = ds.FileSystemFactoryOptions('hive')
    partitioning_factory = ds.HivePartitioning.discover(schema=partition_schema)
    options.partitioning_factory = pickled(partitioning_factory, pickle_module)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    inferred_schema = factory.inspect()
    assert inferred_schema == full_schema
    actual = factory.finish().to_table(columns={'date_int': ds.field("test'; date").cast(pa.int64())})
    assert actual[0][0].as_py() == 1620086400
    partitioning_factory = ds.HivePartitioning.discover(segment_encoding='uri')
    options.partitioning_factory = pickled(partitioning_factory, pickle_module)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals((ds.field("test'; date") == '2021-05-04 00:00:00') & (ds.field("test';[ string'") == '$'))
    partitioning = ds.HivePartitioning(string_partition_schema, segment_encoding='uri')
    options.partitioning = pickled(partitioning, pickle_module)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals((ds.field("test'; date") == '2021-05-04 00:00:00') & (ds.field("test';[ string'") == '$'))
    partitioning_factory = ds.HivePartitioning.discover(segment_encoding='none')
    options.partitioning_factory = pickled(partitioning_factory, pickle_module)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals((ds.field('test%27%3B%20date') == '2021-05-04 00%3A00%3A00') & (ds.field('test%27%3B%5B%20string%27') == '%24'))
    partitioning = ds.HivePartitioning(string_partition_schema_en, segment_encoding='none')
    options.partitioning = pickled(partitioning, pickle_module)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals((ds.field('test%27%3B%20date') == '2021-05-04 00%3A00%3A00') & (ds.field('test%27%3B%5B%20string%27') == '%24'))
    partitioning_factory = ds.HivePartitioning.discover(schema=partition_schema_en, segment_encoding='none')
    options.partitioning_factory = pickled(partitioning_factory, pickle_module)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    with pytest.raises(pa.ArrowInvalid, match='Could not cast segments for partition field'):
        inferred_schema = factory.inspect()