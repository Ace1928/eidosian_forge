import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
@pytest.mark.parametrize('null_obj', [np.nan, None])
def test_unique_null(null_obj, index_or_series_obj):
    obj = index_or_series_obj
    if not allow_na_ops(obj):
        pytest.skip("type doesn't allow for NA operations")
    elif len(obj) < 1:
        pytest.skip("Test doesn't make sense on empty data")
    elif isinstance(obj, pd.MultiIndex):
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")
    values = obj._values
    values[0:2] = null_obj
    klass = type(obj)
    repeated_values = np.repeat(values, range(1, len(values) + 1))
    obj = klass(repeated_values, dtype=obj.dtype)
    result = obj.unique()
    unique_values_raw = dict.fromkeys(obj.values)
    unique_values_not_null = [val for val in unique_values_raw if not pd.isnull(val)]
    unique_values = [null_obj] + unique_values_not_null
    if isinstance(obj, pd.Index):
        expected = pd.Index(unique_values, dtype=obj.dtype)
        if isinstance(obj.dtype, pd.DatetimeTZDtype):
            result = result.normalize()
            expected = expected.normalize()
        tm.assert_index_equal(result, expected, exact=True)
    else:
        expected = np.array(unique_values, dtype=obj.dtype)
        tm.assert_numpy_array_equal(result, expected)