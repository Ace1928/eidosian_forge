import io
import os
import sys
import pytest
import pyarrow as pa
def test_parquet_file_explicitly_closed(tempdir):
    """
    Unopened files should be closed explicitly after use,
    and previously opened files should be left open.
    Applies to read_table, ParquetDataset, and ParquetFile
    """
    fn = tempdir.joinpath('file.parquet')
    table = pa.table({'col1': [0, 1], 'col2': [0, 1]})
    pq.write_table(table, fn)
    with open(fn, 'rb') as f:
        with pq.ParquetFile(f) as p:
            p.read()
            assert not f.closed
            assert not p.closed
        assert not f.closed
        assert not p.closed
    assert f.closed
    assert p.closed
    with pq.ParquetFile(fn) as p:
        p.read()
        assert not p.closed
    assert p.closed