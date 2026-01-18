import pytest
import pandas as pd
from pandas import Categorical
import pandas._testing as tm
@pytest.mark.parametrize('to_replace,value,expected,flip_categories', [(1, 2, [2, 2, 3], False), (1, 4, [4, 2, 3], False), (4, 1, [1, 2, 3], False), (5, 6, [1, 2, 3], False), ([1], 2, [2, 2, 3], False), ([1, 2], 3, [3, 3, 3], False), ([1, 2], 4, [4, 4, 3], False), ((1, 2, 4), 5, [5, 5, 3], False), ((5, 6), 2, [1, 2, 3], False), ([1], [2], [2, 2, 3], False), ([1, 4], [5, 2], [5, 2, 3], False), ([1, 2, 3], [2, 3, 4], [2, 3, 4], False), (1, None, [None, 2, 3], False), (1, pd.NA, [None, 2, 3], False), (3, '4', [1, 2, '4'], False), ([1, 2, '3'], '5', ['5', '5', 3], True)])
@pytest.mark.filterwarnings('ignore:.*with CategoricalDtype is deprecated:FutureWarning')
def test_replace_categorical_series(to_replace, value, expected, flip_categories):
    ser = pd.Series([1, 2, 3], dtype='category')
    result = ser.replace(to_replace, value)
    expected = pd.Series(expected, dtype='category')
    ser.replace(to_replace, value, inplace=True)
    if flip_categories:
        expected = expected.cat.set_categories(expected.cat.categories[::-1])
    tm.assert_series_equal(expected, result, check_category_order=False)
    tm.assert_series_equal(expected, ser, check_category_order=False)