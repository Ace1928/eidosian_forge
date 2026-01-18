from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('df', [pytest.param(pd.DataFrame({'a': ['a', 'b']}), marks=[pytest.mark.filterwarnings('ignore:Applying:DeprecationWarning')])])
def test_pos_object_raises(self, df):
    if np_version_gte1p25:
        with pytest.raises(TypeError, match="^bad operand type for unary \\+: \\'str\\'$"):
            tm.assert_frame_equal(+df, df)
    else:
        tm.assert_series_equal(+df['a'], df['a'])