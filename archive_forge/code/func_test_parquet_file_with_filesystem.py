import io
import os
import sys
import pytest
import pyarrow as pa
@pytest.mark.s3
@pytest.mark.parametrize('use_uri', (True, False))
def test_parquet_file_with_filesystem(s3_example_fs, use_uri):
    s3_fs, s3_uri, s3_path = s3_example_fs
    args = (s3_uri if use_uri else s3_path,)
    kwargs = {} if use_uri else dict(filesystem=s3_fs)
    table = pa.table({'a': range(10)})
    pq.write_table(table, s3_path, filesystem=s3_fs)
    parquet_file = pq.ParquetFile(*args, **kwargs)
    assert parquet_file.read() == table
    assert not parquet_file.closed
    parquet_file.close()
    assert parquet_file.closed
    with pq.ParquetFile(*args, **kwargs) as f:
        assert f.read() == table
        assert not f.closed
    assert f.closed