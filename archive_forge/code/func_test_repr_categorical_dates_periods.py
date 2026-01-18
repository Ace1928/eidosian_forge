from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_categorical_dates_periods(self):
    dt = date_range('2011-01-01 09:00', freq='H', periods=5, tz='US/Eastern')
    p = period_range('2011-01', freq='M', periods=5)
    df = DataFrame({'dt': dt, 'p': p})
    exp = '                         dt        p\n0 2011-01-01 09:00:00-05:00  2011-01\n1 2011-01-01 10:00:00-05:00  2011-02\n2 2011-01-01 11:00:00-05:00  2011-03\n3 2011-01-01 12:00:00-05:00  2011-04\n4 2011-01-01 13:00:00-05:00  2011-05'
    assert repr(df) == exp
    df2 = DataFrame({'dt': Categorical(dt), 'p': Categorical(p)})
    assert repr(df2) == exp