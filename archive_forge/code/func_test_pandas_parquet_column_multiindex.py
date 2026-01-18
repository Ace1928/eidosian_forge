import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_pandas_parquet_column_multiindex(tempdir):
    df = alltypes_sample(size=10)
    df.columns = pd.MultiIndex.from_tuples(list(zip(df.columns, df.columns[::-1])), names=['level_1', 'level_2'])
    filename = tempdir / 'pandas_roundtrip.parquet'
    arrow_table = pa.Table.from_pandas(df)
    assert arrow_table.schema.pandas_metadata is not None
    _write_table(arrow_table, filename)
    table_read = pq.read_pandas(filename)
    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df, df_read)