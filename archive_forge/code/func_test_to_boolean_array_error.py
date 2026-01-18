import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.arrays.boolean import coerce_to_array
@pytest.mark.parametrize('values', [['foo', 'bar'], ['1', '2'], [1, 2], [1.0, 2.0], pd.date_range('20130101', periods=2), np.array(['foo']), np.array([1, 2]), np.array([1.0, 2.0]), [np.nan, {'a': 1}]])
def test_to_boolean_array_error(values):
    msg = 'Need to pass bool-like value'
    with pytest.raises(TypeError, match=msg):
        pd.array(values, dtype='boolean')