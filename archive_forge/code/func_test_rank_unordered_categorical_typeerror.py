from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_rank_unordered_categorical_typeerror():
    cat = pd.Categorical([], ordered=False)
    ser = Series(cat)
    df = ser.to_frame()
    msg = 'Cannot perform rank with non-ordered Categorical'
    gb = ser.groupby(cat)
    with pytest.raises(TypeError, match=msg):
        gb.rank()
    gb2 = df.groupby(cat)
    with pytest.raises(TypeError, match=msg):
        gb2.rank()