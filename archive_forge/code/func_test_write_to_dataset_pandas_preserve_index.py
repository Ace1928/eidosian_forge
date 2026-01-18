import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_write_to_dataset_pandas_preserve_index(tempdir):
    df = pd.DataFrame({'part': ['a', 'a', 'b'], 'col': [1, 2, 3]})
    df.index = pd.Index(['a', 'b', 'c'], name='idx')
    table = pa.table(df)
    df_cat = df[['col', 'part']].copy()
    df_cat['part'] = df_cat['part'].astype('category')
    pq.write_to_dataset(table, str(tempdir / 'case1'), partition_cols=['part'])
    result = pq.read_table(str(tempdir / 'case1')).to_pandas()
    tm.assert_frame_equal(result, df_cat)
    pq.write_to_dataset(table, str(tempdir / 'case2'))
    result = pq.read_table(str(tempdir / 'case2')).to_pandas()
    tm.assert_frame_equal(result, df)
    pq.write_table(table, str(tempdir / 'data.parquet'))
    result = pq.read_table(str(tempdir / 'data.parquet')).to_pandas()
    tm.assert_frame_equal(result, df)