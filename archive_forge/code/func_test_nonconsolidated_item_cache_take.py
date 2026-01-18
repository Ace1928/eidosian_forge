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
def test_nonconsolidated_item_cache_take():
    df = DataFrame()
    df['col1'] = Series(['a'], dtype=object)
    df['col2'] = Series([0], dtype=object)
    df['col1'] == 'A'
    df[df['col2'] == 0]
    df.at[0, 'col1'] = 'A'
    expected = DataFrame({'col1': ['A'], 'col2': [0]}, dtype=object)
    tm.assert_frame_equal(df, expected)
    assert df.at[0, 'col1'] == 'A'