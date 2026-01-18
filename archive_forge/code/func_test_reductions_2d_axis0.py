import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE
@pytest.mark.parametrize('method', ['mean', 'median', 'var', 'std', 'sum', 'prod'])
@pytest.mark.parametrize('min_count', [0, 1])
def test_reductions_2d_axis0(self, data, method, min_count):
    if min_count == 1 and method not in ['sum', 'prod']:
        pytest.skip(f'min_count not relevant for {method}')
    arr2d = data.reshape(1, -1)
    kwargs = {}
    if method in ['std', 'var']:
        kwargs['ddof'] = 0
    elif method in ['prod', 'sum']:
        kwargs['min_count'] = min_count
    try:
        result = getattr(arr2d, method)(axis=0, **kwargs)
    except Exception as err:
        try:
            getattr(data, method)()
        except Exception as err2:
            assert type(err) == type(err2)
            return
        else:
            raise AssertionError('Both reductions should raise or neither')

    def get_reduction_result_dtype(dtype):
        if dtype.itemsize == 8:
            return dtype
        elif dtype.kind in 'ib':
            return NUMPY_INT_TO_DTYPE[np.dtype(int)]
        else:
            return NUMPY_INT_TO_DTYPE[np.dtype('uint')]
    if method in ['sum', 'prod']:
        expected = data
        if data.dtype.kind in 'iub':
            dtype = get_reduction_result_dtype(data.dtype)
            expected = data.astype(dtype)
            assert dtype == expected.dtype
        if min_count == 0:
            fill_value = 1 if method == 'prod' else 0
            expected = expected.fillna(fill_value)
        tm.assert_extension_array_equal(result, expected)
    elif method == 'median':
        expected = data
        tm.assert_extension_array_equal(result, expected)
    elif method in ['mean', 'std', 'var']:
        if is_integer_dtype(data) or is_bool_dtype(data):
            data = data.astype('Float64')
        if method == 'mean':
            tm.assert_extension_array_equal(result, data)
        else:
            tm.assert_extension_array_equal(result, data - data)