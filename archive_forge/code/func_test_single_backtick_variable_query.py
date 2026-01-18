import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_single_backtick_variable_query(self, df):
    res = df.query('1 < `B B`')
    expect = df[1 < df['B B']]
    tm.assert_frame_equal(res, expect)