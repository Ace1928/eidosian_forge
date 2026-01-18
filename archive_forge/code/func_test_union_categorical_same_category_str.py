import numpy as np
import pytest
from pandas.core.dtypes.concat import union_categoricals
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_union_categorical_same_category_str(self):
    c1 = Categorical(['z', 'z', 'z'], categories=['x', 'y', 'z'])
    c2 = Categorical(['x', 'x', 'x'], categories=['x', 'y', 'z'])
    res = union_categoricals([c1, c2])
    exp = Categorical(['z', 'z', 'z', 'x', 'x', 'x'], categories=['x', 'y', 'z'])
    tm.assert_categorical_equal(res, exp)