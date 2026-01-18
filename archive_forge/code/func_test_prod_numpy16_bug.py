import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_prod_numpy16_bug():
    ser = Series([1.0, 1.0, 1.0], index=range(3))
    result = ser.prod()
    assert not isinstance(result, Series)