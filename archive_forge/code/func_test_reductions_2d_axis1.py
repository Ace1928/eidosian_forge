import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE
@pytest.mark.parametrize('method', ['mean', 'median', 'var', 'std', 'sum', 'prod'])
def test_reductions_2d_axis1(self, data, method):
    arr2d = data.reshape(1, -1)
    try:
        result = getattr(arr2d, method)(axis=1)
    except Exception as err:
        try:
            getattr(data, method)()
        except Exception as err2:
            assert type(err) == type(err2)
            return
        else:
            raise AssertionError('Both reductions should raise or neither')
    assert result.shape == (1,)
    expected_scalar = getattr(data, method)()
    res = result[0]
    assert is_matching_na(res, expected_scalar) or res == expected_scalar