import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import FileSystem, LocalFileSystem
@pytest.mark.pandas
@pytest.mark.s3
def test_parquet_writer_filesystem_s3(s3_example_fs):
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)
    fs, uri, path = s3_example_fs
    with pq.ParquetWriter(path, table.schema, filesystem=fs, version='2.6') as writer:
        writer.write_table(table)
    result = _read_table(uri).to_pandas()
    tm.assert_frame_equal(result, df)