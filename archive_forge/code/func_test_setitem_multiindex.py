import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_multiindex(self):
    cols = ['A', 'w', 'l', 'a', 'x', 'X', 'd', 'profit']
    index = MultiIndex.from_product([np.arange(0, 100), np.arange(0, 80)], names=['time', 'firm'])
    t, n = (0, 2)
    df = DataFrame(np.nan, columns=cols, index=index)
    self.check(target=df, indexers=((t, n), 'X'), value=0)
    df = DataFrame(-999, columns=cols, index=index)
    self.check(target=df, indexers=((t, n), 'X'), value=1)
    df = DataFrame(columns=cols, index=index)
    self.check(target=df, indexers=((t, n), 'X'), value=2)
    df = DataFrame(-999, columns=cols, index=index)
    self.check(target=df, indexers=((t, n), 'X'), value=np.array(3), expected=3)