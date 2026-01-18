import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
from pandas.core.arrays import BooleanArray
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
@pytest.mark.parametrize('func', [np.maximum, np.minimum])
def test_numpy_ufuncs_reductions(index, func, request):
    if len(index) == 0:
        pytest.skip("Test doesn't make sense for empty index.")
    if isinstance(index, CategoricalIndex) and index.dtype.ordered is False:
        with pytest.raises(TypeError, match='is not ordered for'):
            func.reduce(index)
        return
    else:
        result = func.reduce(index)
    if func is np.maximum:
        expected = index.max(skipna=False)
    else:
        expected = index.min(skipna=False)
    assert type(result) is type(expected)
    if isna(result):
        assert isna(expected)
    else:
        assert result == expected