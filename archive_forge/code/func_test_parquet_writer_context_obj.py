import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import FileSystem, LocalFileSystem
@pytest.mark.pandas
def test_parquet_writer_context_obj(tempdir):
    df = _test_dataframe(100)
    df['unique_id'] = 0
    arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    out = pa.BufferOutputStream()
    with pq.ParquetWriter(out, arrow_table.schema, version='2.6') as writer:
        frames = []
        for i in range(10):
            df['unique_id'] = i
            arrow_table = pa.Table.from_pandas(df, preserve_index=False)
            writer.write_table(arrow_table)
            frames.append(df.copy())
    buf = out.getvalue()
    result = _read_table(pa.BufferReader(buf))
    expected = pd.concat(frames, ignore_index=True)
    tm.assert_frame_equal(result.to_pandas(), expected)