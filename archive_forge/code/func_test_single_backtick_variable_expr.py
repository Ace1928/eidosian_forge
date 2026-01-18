import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_single_backtick_variable_expr(self, df):
    res = df.eval('A + `B B`')
    expect = df['A'] + df['B B']
    tm.assert_series_equal(res, expect)