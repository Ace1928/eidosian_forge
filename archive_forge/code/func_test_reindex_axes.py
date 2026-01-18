from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_reindex_axes(self):
    df = DataFrame(np.ones((3, 3)), index=[datetime(2012, 1, 1), datetime(2012, 1, 2), datetime(2012, 1, 3)], columns=['a', 'b', 'c'])
    time_freq = date_range('2012-01-01', '2012-01-03', freq='d')
    some_cols = ['a', 'b']
    index_freq = df.reindex(index=time_freq).index.freq
    both_freq = df.reindex(index=time_freq, columns=some_cols).index.freq
    seq_freq = df.reindex(index=time_freq).reindex(columns=some_cols).index.freq
    assert index_freq == both_freq
    assert index_freq == seq_freq