from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_rsplit_to_dataframe_expand(any_string_dtype):
    s = Series(['some_equal_splits', 'with_no_nans'], dtype=any_string_dtype)
    result = s.str.rsplit('_', expand=True)
    exp = DataFrame({0: ['some', 'with'], 1: ['equal', 'no'], 2: ['splits', 'nans']}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, exp)
    result = s.str.rsplit('_', expand=True, n=2)
    exp = DataFrame({0: ['some', 'with'], 1: ['equal', 'no'], 2: ['splits', 'nans']}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, exp)
    result = s.str.rsplit('_', expand=True, n=1)
    exp = DataFrame({0: ['some_equal', 'with_no'], 1: ['splits', 'nans']}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, exp)