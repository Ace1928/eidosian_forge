from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.parametrize('method', ['stack', 'unstack'])
def test_stack_unstack_wrong_level_name(self, method, multiindex_dataframe_random_data, future_stack):
    frame = multiindex_dataframe_random_data
    df = frame.loc['foo']
    kwargs = {'future_stack': future_stack} if method == 'stack' else {}
    with pytest.raises(KeyError, match='does not match index name'):
        getattr(df, method)('mistake', **kwargs)
    if method == 'unstack':
        s = df.iloc[:, 0]
        with pytest.raises(KeyError, match='does not match index name'):
            getattr(s, method)('mistake', **kwargs)