import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_round_builtin(self):
    df = DataFrame({'col1': [1.123, 2.123, 3.123], 'col2': [1.234, 2.234, 3.234]})
    expected_rounded = DataFrame({'col1': [1.0, 2.0, 3.0], 'col2': [1.0, 2.0, 3.0]})
    tm.assert_frame_equal(round(df), expected_rounded)