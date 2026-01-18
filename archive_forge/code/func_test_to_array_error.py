import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
@pytest.mark.parametrize('values', [['foo', 'bar'], 'foo', 1, 1.0, pd.date_range('20130101', periods=2), np.array(['foo']), [[1, 2], [3, 4]], [np.nan, {'a': 1}], np.array([pd.NA] * 6, dtype=object).reshape(3, 2)])
def test_to_array_error(values):
    msg = '|'.join(['cannot be converted to FloatingDtype', 'values must be a 1D list-like', 'Cannot pass scalar', "float\\(\\) argument must be a string or a (real )?number, not 'dict'", "could not convert string to float: 'foo'", "could not convert string to float: np\\.str_\\('foo'\\)"])
    with pytest.raises((TypeError, ValueError), match=msg):
        pd.array(values, dtype='Float64')