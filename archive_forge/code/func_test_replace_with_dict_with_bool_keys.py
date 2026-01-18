from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_with_dict_with_bool_keys(self):
    df = DataFrame({0: [True, False], 1: [False, True]})
    result = df.replace({'asdf': 'asdb', True: 'yes'})
    expected = DataFrame({0: ['yes', False], 1: [False, 'yes']})
    tm.assert_frame_equal(result, expected)