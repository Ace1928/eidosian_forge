from collections import abc
import email
from email.parser import Parser
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_records_timeseries(self):
    index = date_range('1/1/2000', periods=10)
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), index=index, columns=['a', 'b', 'c'])
    result = df.to_records()
    assert result['index'].dtype == 'M8[ns]'
    result = df.to_records(index=False)