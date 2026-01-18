import io
import os
import sys
import pytest
import pyarrow as pa
@pytest.mark.pandas
def test_read_single_row_group():
    N, K = (10000, 4)
    df = alltypes_sample(size=N)
    a_table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    _write_table(a_table, buf, row_group_size=N / K, compression='snappy', version='2.6')
    buf.seek(0)
    pf = pq.ParquetFile(buf)
    assert pf.num_row_groups == K
    row_groups = [pf.read_row_group(i) for i in range(K)]
    result = pa.concat_tables(row_groups)
    tm.assert_frame_equal(df, result.to_pandas())