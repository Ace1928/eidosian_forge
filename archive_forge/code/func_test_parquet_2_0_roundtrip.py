import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
@pytest.mark.parametrize('chunk_size', [None, 1000])
def test_parquet_2_0_roundtrip(tempdir, chunk_size):
    df = alltypes_sample(size=10000, categorical=True)
    filename = tempdir / 'pandas_roundtrip.parquet'
    arrow_table = pa.Table.from_pandas(df)
    assert arrow_table.schema.pandas_metadata is not None
    _write_table(arrow_table, filename, version='2.6', chunk_size=chunk_size)
    table_read = pq.read_pandas(filename)
    assert table_read.schema.pandas_metadata is not None
    read_metadata = table_read.schema.metadata
    assert arrow_table.schema.metadata == read_metadata
    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df, df_read)