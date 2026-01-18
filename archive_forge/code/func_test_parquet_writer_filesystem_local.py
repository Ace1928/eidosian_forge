import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import FileSystem, LocalFileSystem
@pytest.mark.pandas
@pytest.mark.parametrize('filesystem', [None, LocalFileSystem._get_instance(), fs.LocalFileSystem()])
def test_parquet_writer_filesystem_local(tempdir, filesystem):
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)
    path = str(tempdir / 'data.parquet')
    with pq.ParquetWriter(path, table.schema, filesystem=filesystem, version='2.6') as writer:
        writer.write_table(table)
    result = _read_table(path).to_pandas()
    tm.assert_frame_equal(result, df)