import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_set_index_drop_update_column(using_copy_on_write):
    df = DataFrame({'a': [1, 2], 'b': 1.5})
    view = df[:]
    df = df.set_index('a', drop=True)
    expected = df.index.copy(deep=True)
    view.iloc[0, 0] = 100
    tm.assert_index_equal(df.index, expected)