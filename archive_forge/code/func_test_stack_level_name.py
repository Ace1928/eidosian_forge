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
@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_level_name(self, multiindex_dataframe_random_data, future_stack):
    frame = multiindex_dataframe_random_data
    unstacked = frame.unstack('second')
    result = unstacked.stack('exp', future_stack=future_stack)
    expected = frame.unstack().stack(0, future_stack=future_stack)
    tm.assert_frame_equal(result, expected)
    result = frame.stack('exp', future_stack=future_stack)
    expected = frame.stack(future_stack=future_stack)
    tm.assert_series_equal(result, expected)