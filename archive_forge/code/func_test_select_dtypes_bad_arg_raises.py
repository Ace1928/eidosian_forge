import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
def test_select_dtypes_bad_arg_raises(self):
    df = DataFrame({'a': list('abc'), 'g': list('abc'), 'b': list(range(1, 4)), 'c': np.arange(3, 6).astype('u1'), 'd': np.arange(4.0, 7.0, dtype='float64'), 'e': [True, False, True], 'f': pd.date_range('now', periods=3).values})
    msg = 'data type.*not understood'
    with pytest.raises(TypeError, match=msg):
        df.select_dtypes(['blargy, blarg, blarg'])