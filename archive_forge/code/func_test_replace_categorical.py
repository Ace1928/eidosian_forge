import pytest
import pandas as pd
from pandas import Categorical
import pandas._testing as tm
@pytest.mark.parametrize('to_replace, value, result, expected_error_msg', [('b', 'c', ['a', 'c'], 'Categorical.categories are different'), ('c', 'd', ['a', 'b'], None), ('a', 'a', ['a', 'b'], None), ('b', None, ['a', None], 'Categorical.categories length are different')])
def test_replace_categorical(to_replace, value, result, expected_error_msg):
    cat = Categorical(['a', 'b'])
    expected = Categorical(result)
    msg = 'The behavior of Series\\.replace \\(and DataFrame.replace\\) with CategoricalDtype'
    warn = FutureWarning if expected_error_msg is not None else None
    with tm.assert_produces_warning(warn, match=msg):
        result = pd.Series(cat, copy=False).replace(to_replace, value)._values
    tm.assert_categorical_equal(result, expected)
    if to_replace == 'b':
        with pytest.raises(AssertionError, match=expected_error_msg):
            tm.assert_categorical_equal(cat, expected)
    ser = pd.Series(cat, copy=False)
    with tm.assert_produces_warning(warn, match=msg):
        ser.replace(to_replace, value, inplace=True)
    tm.assert_categorical_equal(cat, expected)