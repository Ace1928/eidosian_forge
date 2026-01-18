import io
import os
import sys
import pytest
import pyarrow as pa
@pytest.mark.pandas
def test_pass_separate_metadata():
    df = alltypes_sample(size=10000)
    a_table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    _write_table(a_table, buf, compression='snappy', version='2.6')
    buf.seek(0)
    metadata = pq.read_metadata(buf)
    buf.seek(0)
    fileh = pq.ParquetFile(buf, metadata=metadata)
    tm.assert_frame_equal(df, fileh.read().to_pandas())