from datetime import (
import numpy as np
import pytest
from pandas._libs.algos import (
from pandas import (
import pandas._testing as tm
def test_rank_axis(self):
    df = DataFrame([[2, 1], [4, 3]])
    tm.assert_frame_equal(df.rank(axis=0), df.rank(axis='index'))
    tm.assert_frame_equal(df.rank(axis=1), df.rank(axis='columns'))