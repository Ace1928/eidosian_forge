import numpy as np
import pytest
from pandas.core.dtypes.concat import union_categoricals
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_union_categoricals_sort_false_ordered_true(self):
    c1 = Categorical(['b', 'a'], categories=['b', 'a', 'c'], ordered=True)
    c2 = Categorical(['a', 'c'], categories=['b', 'a', 'c'], ordered=True)
    result = union_categoricals([c1, c2], sort_categories=False)
    expected = Categorical(['b', 'a', 'a', 'c'], categories=['b', 'a', 'c'], ordered=True)
    tm.assert_categorical_equal(result, expected)