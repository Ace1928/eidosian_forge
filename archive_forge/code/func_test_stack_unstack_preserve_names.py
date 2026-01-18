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
def test_stack_unstack_preserve_names(self, multiindex_dataframe_random_data, future_stack):
    frame = multiindex_dataframe_random_data
    unstacked = frame.unstack()
    assert unstacked.index.name == 'first'
    assert unstacked.columns.names == ['exp', 'second']
    restacked = unstacked.stack(future_stack=future_stack)
    assert restacked.index.names == frame.index.names