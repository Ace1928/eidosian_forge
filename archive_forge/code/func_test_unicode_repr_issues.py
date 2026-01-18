import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_unicode_repr_issues(self):
    levels = [Index(['a/σ', 'b/σ', 'c/σ']), Index([0, 1])]
    codes = [np.arange(3).repeat(2), np.tile(np.arange(2), 3)]
    index = MultiIndex(levels=levels, codes=codes)
    repr(index.levels)
    repr(index.get_level_values(1))