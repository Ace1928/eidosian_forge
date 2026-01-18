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
def test_open_dataset_from_s3_with_filesystem_uri(s3_server):
    from pyarrow.fs import FileSystem
    host, port, access_key, secret_key = s3_server['connection']
    bucket = 'theirbucket'
    path = 'nested/folder/data.parquet'
    uri = 's3://{}:{}@{}/{}?scheme=http&endpoint_override={}:{}&allow_bucket_creation=true'.format(access_key, secret_key, bucket, path, host, port)
    fs, path = FileSystem.from_uri(uri)
    assert path == 'theirbucket/nested/folder/data.parquet'
    fs.create_dir(bucket)
    table = pa.table({'a': [1, 2, 3]})
    with fs.open_output_stream(path) as out:
        pq.write_table(table, out)
    dataset = ds.dataset(uri, format='parquet')
    assert dataset.to_table().equals(table)
    template = 's3://{}:{}@{{}}?scheme=http&endpoint_override={}:{}'.format(access_key, secret_key, host, port)
    cases = [('theirbucket/nested/folder/', '/data.parquet'), ('theirbucket/nested/folder', 'data.parquet'), ('theirbucket/nested/', 'folder/data.parquet'), ('theirbucket/nested', 'folder/data.parquet'), ('theirbucket', '/nested/folder/data.parquet'), ('theirbucket', 'nested/folder/data.parquet')]
    for prefix, path in cases:
        uri = template.format(prefix)
        dataset = ds.dataset(path, filesystem=uri, format='parquet')
        assert dataset.to_table().equals(table)
    with pytest.raises(pa.ArrowInvalid, match='Missing bucket name'):
        uri = template.format('/')
        ds.dataset('/theirbucket/nested/folder/data.parquet', filesystem=uri)
    error = 'The path component of the filesystem URI must point to a directory but it has a type: `{}`. The path component is `{}` and the given filesystem URI is `{}`'
    path = 'theirbucket/doesnt/exist'
    uri = template.format(path)
    with pytest.raises(ValueError) as exc:
        ds.dataset('data.parquet', filesystem=uri)
    assert str(exc.value) == error.format('NotFound', path, uri)
    path = 'theirbucket/nested/folder/data.parquet'
    uri = template.format(path)
    with pytest.raises(ValueError) as exc:
        ds.dataset('data.parquet', filesystem=uri)
    assert str(exc.value) == error.format('File', path, uri)