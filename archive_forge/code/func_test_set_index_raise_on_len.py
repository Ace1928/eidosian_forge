from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('box', [Series, Index, np.array, iter, lambda x: MultiIndex.from_arrays([x])], ids=['Series', 'Index', 'np.array', 'iter', 'MultiIndex'])
@pytest.mark.parametrize('length', [4, 6], ids=['too_short', 'too_long'])
@pytest.mark.parametrize('append', [True, False])
@pytest.mark.parametrize('drop', [True, False])
def test_set_index_raise_on_len(self, frame_of_index_cols, box, length, drop, append):
    df = frame_of_index_cols
    values = np.random.default_rng(2).integers(0, 10, (length,))
    msg = 'Length mismatch: Expected 5 rows, received array of length.*'
    with pytest.raises(ValueError, match=msg):
        df.set_index(box(values), drop=drop, append=append)
    with pytest.raises(ValueError, match=msg):
        df.set_index(['A', df.A, box(values)], drop=drop, append=append)