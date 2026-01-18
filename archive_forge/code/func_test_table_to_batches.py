from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.pandas
def test_table_to_batches():
    from pandas.testing import assert_frame_equal
    import pandas as pd
    df1 = pd.DataFrame({'a': list(range(10))})
    df2 = pd.DataFrame({'a': list(range(10, 30))})
    batch1 = pa.RecordBatch.from_pandas(df1, preserve_index=False)
    batch2 = pa.RecordBatch.from_pandas(df2, preserve_index=False)
    table = pa.Table.from_batches([batch1, batch2, batch1])
    expected_df = pd.concat([df1, df2, df1], ignore_index=True)
    batches = table.to_batches()
    assert len(batches) == 3
    assert_frame_equal(pa.Table.from_batches(batches).to_pandas(), expected_df)
    batches = table.to_batches(max_chunksize=15)
    assert list(map(len, batches)) == [10, 15, 5, 10]
    assert_frame_equal(table.to_pandas(), expected_df)
    assert_frame_equal(pa.Table.from_batches(batches).to_pandas(), expected_df)
    table_from_iter = pa.Table.from_batches(iter([batch1, batch2, batch1]))
    assert table.equals(table_from_iter)