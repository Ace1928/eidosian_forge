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
def test_unstack_fill_frame_categorical(self):
    data = Series(['a', 'b', 'c', 'a'], dtype='category')
    data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
    result = data.unstack()
    expected = DataFrame({'a': pd.Categorical(list('axa'), categories=list('abc')), 'b': pd.Categorical(list('bcx'), categories=list('abc'))}, index=list('xyz'))
    tm.assert_frame_equal(result, expected)
    msg = 'Cannot setitem on a Categorical with a new category \\(d\\)'
    with pytest.raises(TypeError, match=msg):
        data.unstack(fill_value='d')
    result = data.unstack(fill_value='c')
    expected = DataFrame({'a': pd.Categorical(list('aca'), categories=list('abc')), 'b': pd.Categorical(list('bcc'), categories=list('abc'))}, index=list('xyz'))
    tm.assert_frame_equal(result, expected)