from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def tests_raises_on_nuisance(test_frame):
    df = test_frame
    df['D'] = 'foo'
    r = df.resample('h')
    result = r[['A', 'B']].mean()
    expected = pd.concat([r.A.mean(), r.B.mean()], axis=1)
    tm.assert_frame_equal(result, expected)
    expected = r[['A', 'B', 'C']].mean()
    msg = re.escape('agg function failed [how->mean,dtype->')
    with pytest.raises(TypeError, match=msg):
        r.mean()
    result = r.mean(numeric_only=True)
    tm.assert_frame_equal(result, expected)