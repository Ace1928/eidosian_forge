import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nonnumeric_suffix(self):
    df = DataFrame({'treatment_placebo': [1.0, 2.0], 'treatment_test': [3.0, 4.0], 'result_placebo': [5.0, 6.0], 'A': ['X1', 'X2']})
    expected = DataFrame({'A': ['X1', 'X2', 'X1', 'X2'], 'colname': ['placebo', 'placebo', 'test', 'test'], 'result': [5.0, 6.0, np.nan, np.nan], 'treatment': [1.0, 2.0, 3.0, 4.0]})
    expected = expected.set_index(['A', 'colname'])
    result = wide_to_long(df, ['result', 'treatment'], i='A', j='colname', suffix='[a-z]+', sep='_')
    tm.assert_frame_equal(result, expected)