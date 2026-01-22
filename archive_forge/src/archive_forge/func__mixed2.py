import operator
import re
import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import (
from pandas.core.computation import expressions as expr
@pytest.fixture
def _mixed2(_frame2):
    return DataFrame({'A': _frame2['A'].copy(), 'B': _frame2['B'].astype('float32'), 'C': _frame2['C'].astype('int64'), 'D': _frame2['D'].astype('int32')})