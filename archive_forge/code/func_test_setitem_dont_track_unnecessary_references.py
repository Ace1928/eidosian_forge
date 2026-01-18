import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_setitem_dont_track_unnecessary_references(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': 1, 'c': 1})
    df['b'] = 100
    arr = get_array(df, 'a')
    df.iloc[0, 0] = 100
    assert np.shares_memory(arr, get_array(df, 'a'))