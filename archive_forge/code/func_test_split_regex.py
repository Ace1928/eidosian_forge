from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_split_regex(any_string_dtype):
    values = Series('xxxjpgzzz.jpg', dtype=any_string_dtype)
    result = values.str.split('\\.jpg', regex=True)
    exp = Series([['xxxjpgzzz', '']])
    tm.assert_series_equal(result, exp)