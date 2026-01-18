import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_pandas_categorical_na_type_row_groups():
    df = pd.DataFrame({'col': [None] * 100, 'int': [1.0] * 100})
    df_category = df.astype({'col': 'category', 'int': 'category'})
    table = pa.Table.from_pandas(df)
    table_cat = pa.Table.from_pandas(df_category)
    buf = pa.BufferOutputStream()
    pq.write_table(table_cat, buf, version='2.6', chunk_size=10)
    result = pq.read_table(buf.getvalue())
    assert result[0].equals(table[0])
    assert result[1].equals(table[1])