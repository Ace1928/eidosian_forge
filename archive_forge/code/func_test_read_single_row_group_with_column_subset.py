import io
import os
import sys
import pytest
import pyarrow as pa
@pytest.mark.pandas
def test_read_single_row_group_with_column_subset():
    N, K = (10000, 4)
    df = alltypes_sample(size=N)
    a_table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    _write_table(a_table, buf, row_group_size=N / K, compression='snappy', version='2.6')
    buf.seek(0)
    pf = pq.ParquetFile(buf)
    cols = list(df.columns[:2])
    row_groups = [pf.read_row_group(i, columns=cols) for i in range(K)]
    result = pa.concat_tables(row_groups)
    tm.assert_frame_equal(df[cols], result.to_pandas())
    row_groups = [pf.read_row_group(i, columns=cols + cols) for i in range(K)]
    result = pa.concat_tables(row_groups)
    tm.assert_frame_equal(df[cols], result.to_pandas())