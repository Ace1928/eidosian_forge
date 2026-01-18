import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ascending', [None, [True, None], [False, 'True']])
def test_sort_index_ascending_bad_value_raises(self, ascending):
    df = DataFrame(np.arange(64))
    length = len(df.index)
    df.index = [(i - length / 2) % length for i in range(length)]
    match = 'For argument "ascending" expected type bool'
    with pytest.raises(ValueError, match=match):
        df.sort_index(axis=0, ascending=ascending, na_position='first')