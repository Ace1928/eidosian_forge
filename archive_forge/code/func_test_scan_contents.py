import io
import os
import sys
import pytest
import pyarrow as pa
@pytest.mark.pandas
def test_scan_contents():
    N, K = (10000, 4)
    df = alltypes_sample(size=N)
    a_table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    _write_table(a_table, buf, row_group_size=N / K, compression='snappy', version='2.6')
    buf.seek(0)
    pf = pq.ParquetFile(buf)
    assert pf.scan_contents() == 10000
    assert pf.scan_contents(df.columns[:4]) == 10000