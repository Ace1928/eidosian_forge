from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('method, exp', [['partition', {0: ['a', 'c', np.nan, 'f', None], 1: ['_', '_', np.nan, '_', None], 2: ['b_c', 'd_e', np.nan, 'g_h', None]}], ['rpartition', {0: ['a_b', 'c_d', np.nan, 'f_g', None], 1: ['_', '_', np.nan, '_', None], 2: ['c', 'e', np.nan, 'h', None]}]])
def test_partition_to_dataframe_from_series(any_string_dtype, method, exp):
    s = Series(['a_b_c', 'c_d_e', np.nan, 'f_g_h', None], dtype=any_string_dtype)
    result = getattr(s.str, method)('_', expand=True)
    expected = DataFrame(exp, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)