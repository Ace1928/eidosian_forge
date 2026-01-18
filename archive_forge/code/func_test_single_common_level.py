import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
def test_single_common_level(self):
    index_left = MultiIndex.from_tuples([('K0', 'X0'), ('K0', 'X1'), ('K1', 'X2')], names=['key', 'X'])
    left = DataFrame({'A': ['A0', 'A1', 'A2'], 'B': ['B0', 'B1', 'B2']}, index=index_left)
    index_right = MultiIndex.from_tuples([('K0', 'Y0'), ('K1', 'Y1'), ('K2', 'Y2'), ('K2', 'Y3')], names=['key', 'Y'])
    right = DataFrame({'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']}, index=index_right)
    result = left.join(right)
    expected = merge(left.reset_index(), right.reset_index(), on=['key'], how='inner').set_index(['key', 'X', 'Y'])
    tm.assert_frame_equal(result, expected)