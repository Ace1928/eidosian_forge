import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_start_with_spaces(self, df):
    res = df.eval('` A` + `  `')
    expect = df[' A'] + df['  ']
    tm.assert_series_equal(res, expect)