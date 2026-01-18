from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_slice_consolidate_invalidate_item_cache(self, using_copy_on_write):
    with option_context('chained_assignment', None):
        df = DataFrame({'aa': np.arange(5), 'bb': [2.2] * 5})
        df['cc'] = 0.0
        df['bb']
        with tm.raises_chained_assignment_error():
            df['bb'].iloc[0] = 0.17
        df._clear_item_cache()
        if not using_copy_on_write:
            tm.assert_almost_equal(df['bb'][0], 0.17)
        else:
            tm.assert_almost_equal(df['bb'][0], 2.2)