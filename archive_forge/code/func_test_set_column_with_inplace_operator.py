import numpy as np
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_set_column_with_inplace_operator(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    with tm.assert_produces_warning(None):
        df['a'] += 1
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    ser = df['a']
    with tm.assert_cow_warning(warn_copy_on_write):
        ser += 1