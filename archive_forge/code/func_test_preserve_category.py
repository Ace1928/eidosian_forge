import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_preserve_category(self):
    data = DataFrame({'A': [1, 2], 'B': pd.Categorical(['X', 'Y'])})
    result = melt(data, ['B'], ['A'])
    expected = DataFrame({'B': pd.Categorical(['X', 'Y']), 'variable': ['A', 'A'], 'value': [1, 2]})
    tm.assert_frame_equal(result, expected)