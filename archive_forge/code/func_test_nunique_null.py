import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops
@pytest.mark.parametrize('null_obj', [np.nan, None])
def test_nunique_null(null_obj, index_or_series_obj):
    obj = index_or_series_obj
    if not allow_na_ops(obj):
        pytest.skip("type doesn't allow for NA operations")
    elif isinstance(obj, pd.MultiIndex):
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")
    values = obj._values
    values[0:2] = null_obj
    klass = type(obj)
    repeated_values = np.repeat(values, range(1, len(values) + 1))
    obj = klass(repeated_values, dtype=obj.dtype)
    if isinstance(obj, pd.CategoricalIndex):
        assert obj.nunique() == len(obj.categories)
        assert obj.nunique(dropna=False) == len(obj.categories) + 1
    else:
        num_unique_values = len(obj.unique())
        assert obj.nunique() == max(0, num_unique_values - 1)
        assert obj.nunique(dropna=False) == max(0, num_unique_values)