import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_required_arguments3(self):
    msg = 'Of the three parameters: start, end, and periods, exactly two must be specified'
    with pytest.raises(ValueError, match=msg):
        period_range(start='2017Q1')
    with pytest.raises(ValueError, match=msg):
        period_range(end='2017Q1')
    with pytest.raises(ValueError, match=msg):
        period_range(periods=5)
    with pytest.raises(ValueError, match=msg):
        period_range()