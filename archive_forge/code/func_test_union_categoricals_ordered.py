import numpy as np
import pytest
from pandas.core.dtypes.concat import union_categoricals
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_union_categoricals_ordered(self):
    c1 = Categorical([1, 2, 3], ordered=True)
    c2 = Categorical([1, 2, 3], ordered=False)
    msg = 'Categorical.ordered must be the same'
    with pytest.raises(TypeError, match=msg):
        union_categoricals([c1, c2])
    res = union_categoricals([c1, c1])
    exp = Categorical([1, 2, 3, 1, 2, 3], ordered=True)
    tm.assert_categorical_equal(res, exp)
    c1 = Categorical([1, 2, 3, np.nan], ordered=True)
    c2 = Categorical([3, 2], categories=[1, 2, 3], ordered=True)
    res = union_categoricals([c1, c2])
    exp = Categorical([1, 2, 3, np.nan, 3, 2], ordered=True)
    tm.assert_categorical_equal(res, exp)
    c1 = Categorical([1, 2, 3], ordered=True)
    c2 = Categorical([1, 2, 3], categories=[3, 2, 1], ordered=True)
    msg = 'to union ordered Categoricals, all categories must be the same'
    with pytest.raises(TypeError, match=msg):
        union_categoricals([c1, c2])