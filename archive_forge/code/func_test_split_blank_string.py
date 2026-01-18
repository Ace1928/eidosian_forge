from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_split_blank_string(any_string_dtype):
    values = Series([''], name='test', dtype=any_string_dtype)
    result = values.str.split(expand=True)
    exp = DataFrame([[]], dtype=any_string_dtype)
    tm.assert_frame_equal(result, exp)