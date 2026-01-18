from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('append', [True, False])
@pytest.mark.parametrize('drop', [True, False])
def test_set_index_pass_multiindex(self, frame_of_index_cols, drop, append):
    df = frame_of_index_cols
    keys = MultiIndex.from_arrays([df['A'], df['B']], names=['A', 'B'])
    result = df.set_index(keys, drop=drop, append=append)
    expected = df.set_index(['A', 'B'], drop=False, append=append)
    tm.assert_frame_equal(result, expected)