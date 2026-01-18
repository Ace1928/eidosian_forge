import numpy as np
import pytest
from pandas.core.dtypes.concat import union_categoricals
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_union_categoricals_sort_false_empty(self):
    c1 = Categorical([])
    c2 = Categorical([])
    result = union_categoricals([c1, c2], sort_categories=False)
    expected = Categorical([])
    tm.assert_categorical_equal(result, expected)