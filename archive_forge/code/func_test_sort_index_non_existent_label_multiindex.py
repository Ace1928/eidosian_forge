import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_non_existent_label_multiindex(self):
    df = DataFrame(0, columns=[], index=MultiIndex.from_product([[], []]))
    with tm.assert_produces_warning(None):
        df.loc['b', '2'] = 1
        df.loc['a', '3'] = 1
    result = df.sort_index().index.is_monotonic_increasing
    assert result is True