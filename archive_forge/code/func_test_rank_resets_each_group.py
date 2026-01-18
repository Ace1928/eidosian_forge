from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('pct,exp', [(False, [3.0, 3.0, 3.0, 3.0, 3.0]), (True, [0.6, 0.6, 0.6, 0.6, 0.6])])
def test_rank_resets_each_group(pct, exp):
    df = DataFrame({'key': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'], 'val': [1] * 10})
    result = df.groupby('key').rank(pct=pct)
    exp_df = DataFrame(exp * 2, columns=['val'])
    tm.assert_frame_equal(result, exp_df)