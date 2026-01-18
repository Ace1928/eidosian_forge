from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('box', [Series, Index, np.array, list, lambda x: MultiIndex.from_arrays([x])])
@pytest.mark.parametrize('append, index_name', [(True, None), (True, 'A'), (True, 'B'), (True, 'test'), (False, None)])
@pytest.mark.parametrize('drop', [True, False])
def test_set_index_pass_arrays(self, frame_of_index_cols, drop, append, index_name, box):
    df = frame_of_index_cols
    df.index.name = index_name
    keys = ['A', box(df['B'])]
    names = ['A', None if box in [np.array, list, tuple, iter] else 'B']
    result = df.set_index(keys, drop=drop, append=append)
    expected = df.set_index(['A', 'B'], drop=False, append=append)
    expected = expected.drop('A', axis=1) if drop else expected
    expected.index.names = [index_name] + names if append else names
    tm.assert_frame_equal(result, expected)