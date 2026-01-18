import datetime
import io
import warnings
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
def test_pandas_parquet_datetime_tz():
    s = pd.Series([datetime.datetime(2017, 9, 6)], dtype='datetime64[us]')
    s = s.dt.tz_localize('utc')
    s.index = s
    df = pd.DataFrame({'tz_aware': s, 'tz_eastern': s.dt.tz_convert('US/Eastern')}, index=s)
    f = io.BytesIO()
    arrow_table = pa.Table.from_pandas(df)
    _write_table(arrow_table, f)
    f.seek(0)
    table_read = pq.read_pandas(f)
    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df, df_read)