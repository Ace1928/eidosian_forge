from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_split_to_dataframe_unequal_splits(any_string_dtype):
    s = Series(['some_unequal_splits', 'one_of_these_things_is_not'], dtype=any_string_dtype)
    result = s.str.split('_', expand=True)
    exp = DataFrame({0: ['some', 'one'], 1: ['unequal', 'of'], 2: ['splits', 'these'], 3: [None, 'things'], 4: [None, 'is'], 5: [None, 'not']}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, exp)