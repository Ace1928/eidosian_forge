import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
from pandas.core.arrays import BooleanArray
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
@pytest.mark.parametrize('func', [np.exp, np.exp2, np.expm1, np.log, np.log2, np.log10, np.log1p, np.sqrt, np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan, np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh, np.deg2rad, np.rad2deg], ids=lambda x: x.__name__)
def test_numpy_ufuncs_basic(index, func):
    if isinstance(index, DatetimeIndexOpsMixin):
        with tm.external_error_raised((TypeError, AttributeError)):
            with np.errstate(all='ignore'):
                func(index)
    elif is_numeric_dtype(index) and (not (is_complex_dtype(index) and func in [np.deg2rad, np.rad2deg])):
        with np.errstate(all='ignore'):
            result = func(index)
            arr_result = func(index.values)
            if arr_result.dtype == np.float16:
                arr_result = arr_result.astype(np.float32)
            exp = Index(arr_result, name=index.name)
        tm.assert_index_equal(result, exp)
        if isinstance(index.dtype, np.dtype) and is_numeric_dtype(index):
            if is_complex_dtype(index):
                assert result.dtype == index.dtype
            elif index.dtype in ['bool', 'int8', 'uint8']:
                assert result.dtype in ['float16', 'float32']
            elif index.dtype in ['int16', 'uint16', 'float32']:
                assert result.dtype == 'float32'
            else:
                assert result.dtype == 'float64'
        else:
            assert type(result) is Index
    elif len(index) == 0:
        pass
    else:
        with tm.external_error_raised((TypeError, AttributeError)):
            with np.errstate(all='ignore'):
                func(index)