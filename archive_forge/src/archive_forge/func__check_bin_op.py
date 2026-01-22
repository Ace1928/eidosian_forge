import operator
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def _check_bin_op(op):
    result = op(df1, df2)
    expected = DataFrame(op(df1.values, df2.values), index=df1.index, columns=df1.columns)
    assert result.values.dtype == np.bool_
    tm.assert_frame_equal(result, expected)