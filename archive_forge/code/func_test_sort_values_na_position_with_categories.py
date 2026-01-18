import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_na_position_with_categories(self):
    categories = ['A', 'B', 'C']
    category_indices = [0, 2, 4]
    list_of_nans = [np.nan, np.nan]
    na_indices = [1, 3]
    na_position_first = 'first'
    na_position_last = 'last'
    column_name = 'c'
    reversed_categories = sorted(categories, reverse=True)
    reversed_category_indices = sorted(category_indices, reverse=True)
    reversed_na_indices = sorted(na_indices)
    df = DataFrame({column_name: Categorical(['A', np.nan, 'B', np.nan, 'C'], categories=categories, ordered=True)})
    result = df.sort_values(by=column_name, ascending=True, na_position=na_position_first)
    expected = DataFrame({column_name: Categorical(list_of_nans + categories, categories=categories, ordered=True)}, index=na_indices + category_indices)
    tm.assert_frame_equal(result, expected)
    result = df.sort_values(by=column_name, ascending=True, na_position=na_position_last)
    expected = DataFrame({column_name: Categorical(categories + list_of_nans, categories=categories, ordered=True)}, index=category_indices + na_indices)
    tm.assert_frame_equal(result, expected)
    result = df.sort_values(by=column_name, ascending=False, na_position=na_position_first)
    expected = DataFrame({column_name: Categorical(list_of_nans + reversed_categories, categories=categories, ordered=True)}, index=reversed_na_indices + reversed_category_indices)
    tm.assert_frame_equal(result, expected)
    result = df.sort_values(by=column_name, ascending=False, na_position=na_position_last)
    expected = DataFrame({column_name: Categorical(reversed_categories + list_of_nans, categories=categories, ordered=True)}, index=reversed_category_indices + reversed_na_indices)
    tm.assert_frame_equal(result, expected)