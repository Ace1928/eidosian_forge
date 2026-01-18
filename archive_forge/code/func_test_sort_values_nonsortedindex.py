import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_values_nonsortedindex(self):
    rng = date_range('2011-01-01', '2012-01-01', freq='W')
    ts = DataFrame({'A': np.random.default_rng(2).standard_normal(len(rng)), 'B': np.random.default_rng(2).standard_normal(len(rng))}, index=rng)
    decreasing = ts.sort_values('A', ascending=False)
    msg = 'truncate requires a sorted index'
    with pytest.raises(ValueError, match=msg):
        decreasing.truncate(before='2011-11', after='2011-12')