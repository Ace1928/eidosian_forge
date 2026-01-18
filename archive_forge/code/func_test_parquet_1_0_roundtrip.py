import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
def test_parquet_1_0_roundtrip(tempdir):
    size = 10000
    np.random.seed(0)
    df = pd.DataFrame({'uint8': np.arange(size, dtype=np.uint8), 'uint16': np.arange(size, dtype=np.uint16), 'uint32': np.arange(size, dtype=np.uint32), 'uint64': np.arange(size, dtype=np.uint64), 'int8': np.arange(size, dtype=np.int16), 'int16': np.arange(size, dtype=np.int16), 'int32': np.arange(size, dtype=np.int32), 'int64': np.arange(size, dtype=np.int64), 'float32': np.arange(size, dtype=np.float32), 'float64': np.arange(size, dtype=np.float64), 'bool': np.random.randn(size) > 0, 'str': [str(x) for x in range(size)], 'str_with_nulls': [None] + [str(x) for x in range(size - 2)] + [None], 'empty_str': [''] * size})
    filename = tempdir / 'pandas_roundtrip.parquet'
    arrow_table = pa.Table.from_pandas(df)
    _write_table(arrow_table, filename, version='1.0')
    table_read = _read_table(filename)
    df_read = table_read.to_pandas()
    df['uint32'] = df['uint32'].values.astype(np.int64)
    tm.assert_frame_equal(df, df_read)