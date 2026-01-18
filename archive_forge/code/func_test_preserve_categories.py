from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_preserve_categories():
    categories = list('abc')
    df = DataFrame({'A': Categorical(list('ba'), categories=categories, ordered=True)})
    sort_index = CategoricalIndex(categories, categories, ordered=True, name='A')
    nosort_index = CategoricalIndex(list('bac'), categories, ordered=True, name='A')
    tm.assert_index_equal(df.groupby('A', sort=True, observed=False).first().index, sort_index)
    tm.assert_index_equal(df.groupby('A', sort=False, observed=False).first().index, nosort_index)
    df = DataFrame({'A': Categorical(list('ba'), categories=categories, ordered=False)})
    sort_index = CategoricalIndex(categories, categories, ordered=False, name='A')
    nosort_index = CategoricalIndex(list('bac'), list('abc'), ordered=False, name='A')
    tm.assert_index_equal(df.groupby('A', sort=True, observed=False).first().index, sort_index)
    tm.assert_index_equal(df.groupby('A', sort=False, observed=False).first().index, nosort_index)