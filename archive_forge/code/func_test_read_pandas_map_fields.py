import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_read_pandas_map_fields(tempdir):
    df = pd.DataFrame({'col1': pd.Series([[('id', 'something'), ('value2', 'else')], [('id', 'something2'), ('value', 'else2')]]), 'col2': pd.Series(['foo', 'bar'])})
    filename = tempdir / 'data.parquet'
    udt = pa.map_(pa.string(), pa.string())
    schema = pa.schema([pa.field('col1', udt), pa.field('col2', pa.string())])
    arrow_table = pa.Table.from_pandas(df, schema)
    _write_table(arrow_table, filename)
    result = pq.read_pandas(filename).to_pandas()
    tm.assert_frame_equal(result, df)