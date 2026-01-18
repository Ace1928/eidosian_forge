from datetime import (
import numpy as np
import pytest
from pandas._libs.algos import (
from pandas import (
import pandas._testing as tm
def test_rank_both_inf(self):
    df = DataFrame({'a': [-np.inf, 0, np.inf]})
    expected = DataFrame({'a': [1.0, 2.0, 3.0]})
    result = df.rank()
    tm.assert_frame_equal(result, expected)