from datetime import (
import itertools
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
@pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
def test_strange_column_corruption_issue(self, using_copy_on_write):
    df = DataFrame(index=[0, 1])
    df[0] = np.nan
    wasCol = {}
    with tm.assert_produces_warning(PerformanceWarning, raise_on_extra_warnings=False):
        for i, dt in enumerate(df.index):
            for col in range(100, 200):
                if col not in wasCol:
                    wasCol[col] = 1
                    df[col] = np.nan
                if using_copy_on_write:
                    df.loc[dt, col] = i
                else:
                    df[col][dt] = i
    myid = 100
    first = len(df.loc[pd.isna(df[myid]), [myid]])
    second = len(df.loc[pd.isna(df[myid]), [myid]])
    assert first == second == 0