import numpy as np
import pytest
from pandas.core.dtypes.concat import union_categoricals
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val', [[], ['1']])
def test_union_categoricals_empty(self, val, request, using_infer_string):
    if using_infer_string and val == ['1']:
        request.applymarker(pytest.mark.xfail('object and strings dont match'))
    res = union_categoricals([Categorical([]), Categorical(val)])
    exp = Categorical(val)
    tm.assert_categorical_equal(res, exp)