import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_extension_other(self, frame_or_series):
    obj = frame_or_series(pd.array([1, 2, 3], dtype='Int64'))
    result = obj.replace('', '')
    tm.assert_equal(obj, result)