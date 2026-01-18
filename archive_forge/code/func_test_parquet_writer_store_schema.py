import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import FileSystem, LocalFileSystem
def test_parquet_writer_store_schema(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    path1 = tempdir / 'test_with_schema.parquet'
    with pq.ParquetWriter(path1, table.schema) as writer:
        writer.write_table(table)
    meta = pq.read_metadata(path1)
    assert b'ARROW:schema' in meta.metadata
    assert meta.metadata[b'ARROW:schema']
    path2 = tempdir / 'test_without_schema.parquet'
    with pq.ParquetWriter(path2, table.schema, store_schema=False) as writer:
        writer.write_table(table)
    meta = pq.read_metadata(path2)
    assert meta.metadata is None