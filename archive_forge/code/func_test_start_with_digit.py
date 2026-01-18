import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_start_with_digit(self, df):
    res = df.eval('A + `1e1`')
    expect = df['A'] + df['1e1']
    tm.assert_series_equal(res, expect)