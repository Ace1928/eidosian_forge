import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_reductions_td64_with_nat():
    ser = Series([0, pd.NaT], dtype='m8[ns]')
    exp = ser[0]
    assert ser.median() == exp
    assert ser.min() == exp
    assert ser.max() == exp