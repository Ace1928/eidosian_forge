import numpy as np
import pytest
from pandas.compat import PY311
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_levels(idx):
    levels = idx.levels
    new_levels = [[lev + 'a' for lev in level] for level in levels]
    ind2 = idx.set_levels(new_levels)
    assert_matching(ind2.levels, new_levels)
    assert_matching(idx.levels, levels)
    ind2 = idx.set_levels(new_levels[0], level=0)
    assert_matching(ind2.levels, [new_levels[0], levels[1]])
    assert_matching(idx.levels, levels)
    ind2 = idx.set_levels(new_levels[1], level=1)
    assert_matching(ind2.levels, [levels[0], new_levels[1]])
    assert_matching(idx.levels, levels)
    ind2 = idx.set_levels(new_levels, level=[0, 1])
    assert_matching(ind2.levels, new_levels)
    assert_matching(idx.levels, levels)
    original_index = idx.copy()
    with pytest.raises(ValueError, match='^On'):
        idx.set_levels(['c'], level=0)
    assert_matching(idx.levels, original_index.levels, check_dtype=True)
    with pytest.raises(ValueError, match='^On'):
        idx.set_codes([0, 1, 2, 3, 4, 5], level=0)
    assert_matching(idx.codes, original_index.codes, check_dtype=True)
    with pytest.raises(TypeError, match='^Levels'):
        idx.set_levels('c', level=0)
    assert_matching(idx.levels, original_index.levels, check_dtype=True)
    with pytest.raises(TypeError, match='^Codes'):
        idx.set_codes(1, level=0)
    assert_matching(idx.codes, original_index.codes, check_dtype=True)