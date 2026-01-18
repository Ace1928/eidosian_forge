from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('keys', ['A', 'C', ['A', 'B'], ('tuple', 'as', 'label')])
@pytest.mark.parametrize('drop', [True, False])
def test_set_index_append_to_multiindex(self, frame_of_index_cols, drop, keys):
    df = frame_of_index_cols.set_index(['D'], drop=drop, append=True)
    keys = keys if isinstance(keys, list) else [keys]
    expected = frame_of_index_cols.set_index(['D'] + keys, drop=drop, append=True)
    result = df.set_index(keys, drop=drop, append=True)
    tm.assert_frame_equal(result, expected)