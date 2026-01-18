import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_list_method(self):
    df = tm.SubclassedDataFrame({'A': [0, 1, 2]})
    msg = "The 'method' keyword in SubclassedDataFrame.replace is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg, raise_on_extra_warnings=False):
        result = df.replace([1, 2], method='ffill')
    expected = tm.SubclassedDataFrame({'A': [0, 0, 0]})
    assert isinstance(result, tm.SubclassedDataFrame)
    tm.assert_frame_equal(result, expected)