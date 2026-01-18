import datetime
import io
import warnings
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
def test_noncoerced_nanoseconds_written_without_exception(tempdir):
    n = 9
    df = pd.DataFrame({'x': range(n)}, index=pd.date_range('2017-01-01', freq='1n', periods=n))
    tb = pa.Table.from_pandas(df)
    filename = tempdir / 'written.parquet'
    try:
        pq.write_table(tb, filename, version='2.6')
    except Exception:
        pass
    assert filename.exists()
    recovered_table = pq.read_table(filename)
    assert tb.equals(recovered_table)
    filename = tempdir / 'not_written.parquet'
    with pytest.raises(ValueError):
        pq.write_table(tb, filename, coerce_timestamps='ms', version='2.6')