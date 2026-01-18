import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
@pytest.mark.parametrize('dtype', [str, 'str', np.bytes_, 'S1', np.str_, 'U1'])
@pytest.mark.parametrize('arg', ['include', 'exclude'])
def test_select_dtypes_str_raises(self, dtype, arg):
    df = DataFrame({'a': list('abc'), 'g': list('abc'), 'b': list(range(1, 4)), 'c': np.arange(3, 6).astype('u1'), 'd': np.arange(4.0, 7.0, dtype='float64'), 'e': [True, False, True], 'f': pd.date_range('now', periods=3).values})
    msg = 'string dtypes are not allowed'
    kwargs = {arg: [dtype]}
    with pytest.raises(TypeError, match=msg):
        df.select_dtypes(**kwargs)