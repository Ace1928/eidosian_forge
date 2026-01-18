import datetime as dt
from datetime import date
import re
import numpy as np
import pytest
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_week_of_month_frequency(self):
    d1 = date(2002, 9, 1)
    d2 = date(2013, 10, 27)
    d3 = date(2012, 9, 30)
    idx1 = DatetimeIndex([d1, d2])
    idx2 = DatetimeIndex([d3])
    result_append = idx1.append(idx2)
    expected = DatetimeIndex([d1, d2, d3])
    tm.assert_index_equal(result_append, expected)
    result_union = idx1.union(idx2)
    expected = DatetimeIndex([d1, d3, d2])
    tm.assert_index_equal(result_union, expected)