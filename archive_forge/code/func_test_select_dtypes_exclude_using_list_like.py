import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
def test_select_dtypes_exclude_using_list_like(self):
    df = DataFrame({'a': list('abc'), 'b': list(range(1, 4)), 'c': np.arange(3, 6).astype('u1'), 'd': np.arange(4.0, 7.0, dtype='float64'), 'e': [True, False, True]})
    re = df.select_dtypes(exclude=[np.number])
    ee = df[['a', 'e']]
    tm.assert_frame_equal(re, ee)