import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_read_pandas_column_subset():
    df = _test_dataframe(10000)
    arrow_table = pa.Table.from_pandas(df)
    imos = pa.BufferOutputStream()
    _write_table(arrow_table, imos, version='2.6')
    buf = imos.getvalue()
    reader = pa.BufferReader(buf)
    df_read = pq.read_pandas(reader, columns=['strings', 'uint8']).to_pandas()
    tm.assert_frame_equal(df[['strings', 'uint8']], df_read)