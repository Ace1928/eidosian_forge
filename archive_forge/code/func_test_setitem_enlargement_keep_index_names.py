import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_enlargement_keep_index_names(self):
    mi = MultiIndex.from_tuples([(1, 2, 3)], names=['i1', 'i2', 'i3'])
    df = DataFrame(data=[[10, 20, 30]], index=mi, columns=['A', 'B', 'C'])
    df.loc[0, 0, 0] = df.loc[1, 2, 3]
    mi_expected = MultiIndex.from_tuples([(1, 2, 3), (0, 0, 0)], names=['i1', 'i2', 'i3'])
    expected = DataFrame(data=[[10, 20, 30], [10, 20, 30]], index=mi_expected, columns=['A', 'B', 'C'])
    tm.assert_frame_equal(df, expected)