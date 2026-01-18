import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
def test_select_dtypes_no_view(self):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df_orig = df.copy()
    result = df.select_dtypes(include=['number'])
    result.iloc[0, 0] = 0
    tm.assert_frame_equal(df, df_orig)