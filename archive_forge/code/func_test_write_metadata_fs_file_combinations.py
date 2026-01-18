import datetime
import decimal
from collections import OrderedDict
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip, make_sample_file
from pyarrow.fs import LocalFileSystem
from pyarrow.tests import util
@pytest.mark.s3
def test_write_metadata_fs_file_combinations(tempdir, s3_example_s3fs):
    s3_fs, s3_path = s3_example_s3fs
    meta1 = tempdir / 'meta1'
    meta2 = tempdir / 'meta2'
    meta3 = tempdir / 'meta3'
    meta4 = tempdir / 'meta4'
    meta5 = f'{s3_path}/meta5'
    table = pa.table({'col': range(5)})
    pq.write_metadata(table.schema, meta1, [])
    pq.write_metadata(table.schema, meta2, [], filesystem=LocalFileSystem())
    pq.write_metadata(table.schema, meta3.as_uri(), [])
    with meta4.open('wb+') as meta4_stream:
        pq.write_metadata(table.schema, meta4_stream, [])
    pq.write_metadata(table.schema, meta5, [], filesystem=s3_fs)
    assert meta1.read_bytes() == meta2.read_bytes() == meta3.read_bytes() == meta4.read_bytes() == s3_fs.open(meta5).read()