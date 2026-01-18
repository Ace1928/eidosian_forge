import numpy as np
import pytest
from pandas.core.dtypes.concat import union_categoricals
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_union_categorical_ordered_appearance(self):
    s = Categorical(['x', 'y', 'z'])
    s2 = Categorical(['a', 'b', 'c'])
    result = union_categoricals([s, s2])
    expected = Categorical(['x', 'y', 'z', 'a', 'b', 'c'], categories=['x', 'y', 'z', 'a', 'b', 'c'])
    tm.assert_categorical_equal(result, expected)