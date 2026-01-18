from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('flag', [False, True])
@pytest.mark.parametrize('allow_duplicates', [False, True])
def test_reset_index_duplicate_columns_allow(self, multiindex_df, flag, allow_duplicates):
    df = multiindex_df.rename_axis('A')
    df = df.set_flags(allows_duplicate_labels=flag)
    if flag and allow_duplicates:
        result = df.reset_index(allow_duplicates=allow_duplicates)
        levels = [['A', ''], ['A', ''], ['B', 'b']]
        expected = DataFrame([[0, 0, 2], [1, 1, 3]], columns=MultiIndex.from_tuples(levels))
        tm.assert_frame_equal(result, expected)
    else:
        if not flag and allow_duplicates:
            msg = "Cannot specify 'allow_duplicates=True' when 'self.flags.allows_duplicate_labels' is False"
        else:
            msg = "cannot insert \\('A', ''\\), already exists"
        with pytest.raises(ValueError, match=msg):
            df.reset_index(allow_duplicates=allow_duplicates)