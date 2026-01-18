import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_query_python(self, df, expected1, expected2):
    result = df.query('A>0', engine='python')
    tm.assert_frame_equal(result, expected1)
    result = df.eval('A+1', engine='python')
    tm.assert_series_equal(result, expected2, check_names=False)