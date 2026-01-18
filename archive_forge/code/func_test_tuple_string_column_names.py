import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
def test_tuple_string_column_names():
    mi = MultiIndex.from_tuples([('a', 'aa'), ('a', 'ab'), ('b', 'ba'), ('b', 'bb')])
    df = DataFrame([range(4), range(1, 5), range(2, 6)], columns=mi)
    df['single_index'] = 0
    df_flat = df.copy()
    df_flat.columns = df_flat.columns.to_flat_index()
    df_flat['new_single_index'] = 0
    result = df_flat[[('a', 'aa'), 'new_single_index']]
    expected = DataFrame([[0, 0], [1, 0], [2, 0]], columns=Index([('a', 'aa'), 'new_single_index']))
    tm.assert_frame_equal(result, expected)