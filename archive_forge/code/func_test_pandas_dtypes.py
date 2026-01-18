import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('col', [pd.Series(date_range('2010', periods=5, tz='US/Pacific')), pd.Series(['a', 'b', 'c', 'a', 'd'], dtype='category'), pd.Series([0, 1, 0, 0, 0])])
def test_pandas_dtypes(self, col):
    df = DataFrame({'klass': range(5), 'col': col, 'attr1': [1, 0, 0, 0, 0], 'attr2': col})
    expected_value = pd.concat([pd.Series([1, 0, 0, 0, 0]), col], ignore_index=True)
    result = melt(df, id_vars=['klass', 'col'], var_name='attribute', value_name='value')
    expected = DataFrame({0: list(range(5)) * 2, 1: pd.concat([col] * 2, ignore_index=True), 2: ['attr1'] * 5 + ['attr2'] * 5, 3: expected_value})
    expected.columns = ['klass', 'col', 'attribute', 'value']
    tm.assert_frame_equal(result, expected)