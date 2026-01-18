import numpy as np
import pytest
from pandas.core.dtypes.concat import union_categoricals
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_union_categoricals_nan(self):
    res = union_categoricals([Categorical([1, 2, np.nan]), Categorical([3, 2, np.nan])])
    exp = Categorical([1, 2, np.nan, 3, 2, np.nan])
    tm.assert_categorical_equal(res, exp)
    res = union_categoricals([Categorical(['A', 'B']), Categorical(['B', 'B', np.nan])])
    exp = Categorical(['A', 'B', 'B', 'B', np.nan])
    tm.assert_categorical_equal(res, exp)
    val1 = [pd.Timestamp('2011-01-01'), pd.Timestamp('2011-03-01'), pd.NaT]
    val2 = [pd.NaT, pd.Timestamp('2011-01-01'), pd.Timestamp('2011-02-01')]
    res = union_categoricals([Categorical(val1), Categorical(val2)])
    exp = Categorical(val1 + val2, categories=[pd.Timestamp('2011-01-01'), pd.Timestamp('2011-03-01'), pd.Timestamp('2011-02-01')])
    tm.assert_categorical_equal(res, exp)
    res = union_categoricals([Categorical(np.array([np.nan, np.nan], dtype=object)), Categorical(['X'], categories=pd.Index(['X'], dtype=object))])
    exp = Categorical([np.nan, np.nan, 'X'])
    tm.assert_categorical_equal(res, exp)
    res = union_categoricals([Categorical([np.nan, np.nan]), Categorical([np.nan, np.nan])])
    exp = Categorical([np.nan, np.nan, np.nan, np.nan])
    tm.assert_categorical_equal(res, exp)