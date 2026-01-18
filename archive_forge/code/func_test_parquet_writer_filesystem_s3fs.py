import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import FileSystem, LocalFileSystem
@pytest.mark.pandas
@pytest.mark.s3
def test_parquet_writer_filesystem_s3fs(s3_example_s3fs):
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)
    fs, directory = s3_example_s3fs
    path = directory + '/test.parquet'
    with pq.ParquetWriter(path, table.schema, filesystem=fs, version='2.6') as writer:
        writer.write_table(table)
    result = _read_table(path, filesystem=fs).to_pandas()
    tm.assert_frame_equal(result, df)