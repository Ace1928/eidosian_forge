from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def tests_value_counts_index_names_category_column():
    df = DataFrame({'gender': ['female'], 'country': ['US']})
    df['gender'] = df['gender'].astype('category')
    result = df.groupby('country')['gender'].value_counts()
    df_mi_expected = DataFrame([['US', 'female']], columns=['country', 'gender'])
    df_mi_expected['gender'] = df_mi_expected['gender'].astype('category')
    mi_expected = MultiIndex.from_frame(df_mi_expected)
    expected = Series([1], index=mi_expected, name='count')
    tm.assert_series_equal(result, expected)