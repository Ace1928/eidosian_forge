import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_where_infers_type_instead_of_trying_to_convert_string_to_float(self):
    index = Index([1, np.nan])
    cond = index.notna()
    other = Index(['a', 'b'], dtype='string')
    expected = Index([1.0, 'b'])
    result = index.where(cond, other)
    tm.assert_index_equal(result, expected)