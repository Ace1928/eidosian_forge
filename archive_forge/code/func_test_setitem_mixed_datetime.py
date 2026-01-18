from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_mixed_datetime(self):
    expected = DataFrame({'a': [0, 0, 0, 0, 13, 14], 'b': [datetime(2012, 1, 1), 1, 'x', 'y', datetime(2013, 1, 1), datetime(2014, 1, 1)]})
    df = DataFrame(0, columns=list('ab'), index=range(6))
    df['b'] = pd.NaT
    df.loc[0, 'b'] = datetime(2012, 1, 1)
    with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
        df.loc[1, 'b'] = 1
    df.loc[[2, 3], 'b'] = ('x', 'y')
    A = np.array([[13, np.datetime64('2013-01-01T00:00:00')], [14, np.datetime64('2014-01-01T00:00:00')]])
    df.loc[[4, 5], ['a', 'b']] = A
    tm.assert_frame_equal(df, expected)