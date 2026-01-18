from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_multiple_bug(self, future_stack):
    id_col = [1] * 3 + [2] * 3
    name = ['a'] * 3 + ['b'] * 3
    date = pd.to_datetime(['2013-01-03', '2013-01-04', '2013-01-05'] * 2)
    var1 = np.random.default_rng(2).integers(0, 100, 6)
    df = DataFrame({'ID': id_col, 'NAME': name, 'DATE': date, 'VAR1': var1})
    multi = df.set_index(['DATE', 'ID'])
    multi.columns.name = 'Params'
    unst = multi.unstack('ID')
    msg = re.escape('agg function failed [how->mean,dtype->')
    with pytest.raises(TypeError, match=msg):
        unst.resample('W-THU').mean()
    down = unst.resample('W-THU').mean(numeric_only=True)
    rs = down.stack('ID', future_stack=future_stack)
    xp = unst.loc[:, ['VAR1']].resample('W-THU').mean().stack('ID', future_stack=future_stack)
    xp.columns.name = 'Params'
    tm.assert_frame_equal(rs, xp)