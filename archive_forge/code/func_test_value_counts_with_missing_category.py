import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_value_counts_with_missing_category():
    df = pd.DataFrame({'a': pd.Categorical([1, 2, 4], categories=[1, 2, 3, 4])})
    result = df.value_counts()
    expected = pd.Series([1, 1, 1, 0], index=pd.MultiIndex.from_arrays([pd.CategoricalIndex([1, 2, 4, 3], categories=[1, 2, 3, 4], name='a')]), name='count')
    tm.assert_series_equal(result, expected)