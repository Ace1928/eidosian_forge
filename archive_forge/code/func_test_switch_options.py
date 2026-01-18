import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.single_cpu
@td.skip_array_manager_invalid_test
def test_switch_options():
    with pd.option_context('mode.copy_on_write', False):
        df = DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]})
        subset = df[:]
        subset.iloc[0, 0] = 0
        assert df.iloc[0, 0] == 0
        pd.options.mode.copy_on_write = True
        df = DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]})
        subset = df[:]
        subset.iloc[0, 0] = 0
        assert df.iloc[0, 0] == 1
        pd.options.mode.copy_on_write = False
        df = DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]})
        subset = df[:]
        subset.iloc[0, 0] = 0
        assert df.iloc[0, 0] == 0