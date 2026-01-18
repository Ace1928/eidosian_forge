import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_integer
from pandas.core.arrays import IntegerArray
from pandas.core.arrays.integer import (
@pytest.mark.parametrize('values', [['foo', 'bar'], 'foo', 1, 1.0, pd.date_range('20130101', periods=2), np.array(['foo']), [[1, 2], [3, 4]], [np.nan, {'a': 1}]])
def test_to_integer_array_error(values):
    msg = '|'.join(['cannot be converted to IntegerDtype', 'invalid literal for int\\(\\) with base 10:', 'values must be a 1D list-like', 'Cannot pass scalar', 'int\\(\\) argument must be a string'])
    with pytest.raises((ValueError, TypeError), match=msg):
        pd.array(values, dtype='Int64')
    with pytest.raises((ValueError, TypeError), match=msg):
        IntegerArray._from_sequence(values)