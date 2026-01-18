from datetime import datetime
from io import StringIO
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
from pandas.io.common import get_handle
def test_to_csv_unicode_index(self):
    buf = StringIO()
    s = Series(['א', 'd2'], index=['א', 'ב'])
    s.to_csv(buf, encoding='UTF-8', header=False)
    buf.seek(0)
    s2 = self.read_csv(buf, index_col=0, encoding='UTF-8')
    tm.assert_series_equal(s, s2)