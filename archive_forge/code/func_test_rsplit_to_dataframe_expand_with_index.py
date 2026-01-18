from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_rsplit_to_dataframe_expand_with_index(any_string_dtype):
    s = Series(['some_splits', 'with_index'], index=['preserve', 'me'], dtype=any_string_dtype)
    result = s.str.rsplit('_', expand=True)
    exp = DataFrame({0: ['some', 'with'], 1: ['splits', 'index']}, index=['preserve', 'me'], dtype=any_string_dtype)
    tm.assert_frame_equal(result, exp)