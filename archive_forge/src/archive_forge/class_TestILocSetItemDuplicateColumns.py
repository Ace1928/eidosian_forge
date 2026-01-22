from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
class TestILocSetItemDuplicateColumns:

    def test_iloc_setitem_scalar_duplicate_columns(self):
        df1 = DataFrame([{'A': None, 'B': 1}, {'A': 2, 'B': 2}])
        df2 = DataFrame([{'A': 3, 'B': 3}, {'A': 4, 'B': 4}])
        df = concat([df1, df2], axis=1)
        df.iloc[0, 0] = -1
        assert df.iloc[0, 0] == -1
        assert df.iloc[0, 2] == 3
        assert df.dtypes.iloc[2] == np.int64

    def test_iloc_setitem_list_duplicate_columns(self):
        df = DataFrame([[0, 'str', 'str2']], columns=['a', 'b', 'b'])
        df.iloc[:, 2] = ['str3']
        expected = DataFrame([[0, 'str', 'str3']], columns=['a', 'b', 'b'])
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_series_duplicate_columns(self):
        df = DataFrame(np.arange(8, dtype=np.int64).reshape(2, 4), columns=['A', 'B', 'A', 'B'])
        df.iloc[:, 0] = df.iloc[:, 0].astype(np.float64)
        assert df.dtypes.iloc[2] == np.int64

    @pytest.mark.parametrize(['dtypes', 'init_value', 'expected_value'], [('int64', '0', 0), ('float', '1.2', 1.2)])
    def test_iloc_setitem_dtypes_duplicate_columns(self, dtypes, init_value, expected_value):
        df = DataFrame([[init_value, 'str', 'str2']], columns=['a', 'b', 'b'], dtype=object)
        df.iloc[:, 0] = df.iloc[:, 0].astype(dtypes)
        expected_df = DataFrame([[expected_value, 'str', 'str2']], columns=['a', 'b', 'b'], dtype=object)
        tm.assert_frame_equal(df, expected_df)