from datetime import (
import itertools
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
def test_stale_cached_series_bug_473(self, using_copy_on_write, warn_copy_on_write):
    with option_context('chained_assignment', None):
        Y = DataFrame(np.random.default_rng(2).random((4, 4)), index=('a', 'b', 'c', 'd'), columns=('e', 'f', 'g', 'h'))
        repr(Y)
        Y['e'] = Y['e'].astype('object')
        with tm.raises_chained_assignment_error():
            Y['g']['c'] = np.nan
        repr(Y)
        Y.sum()
        Y['g'].sum()
        if using_copy_on_write:
            assert not pd.isna(Y['g']['c'])
        else:
            assert pd.isna(Y['g']['c'])