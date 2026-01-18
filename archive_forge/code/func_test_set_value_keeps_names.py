import numpy as np
import pytest
from pandas.compat import PY311
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_value_keeps_names():
    lev1 = ['hans', 'hans', 'hans', 'grethe', 'grethe', 'grethe']
    lev2 = ['1', '2', '3'] * 2
    idx = MultiIndex.from_arrays([lev1, lev2], names=['Name', 'Number'])
    df = pd.DataFrame(np.random.default_rng(2).standard_normal((6, 4)), columns=['one', 'two', 'three', 'four'], index=idx)
    df = df.sort_index()
    assert df._is_copy is None
    assert df.index.names == ('Name', 'Number')
    df.at[('grethe', '4'), 'one'] = 99.34
    assert df._is_copy is None
    assert df.index.names == ('Name', 'Number')