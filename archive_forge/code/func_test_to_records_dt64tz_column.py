from collections import abc
import email
from email.parser import Parser
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_records_dt64tz_column(self):
    df = DataFrame({'A': date_range('2012-01-01', '2012-01-02', tz='US/Eastern')})
    result = df.to_records()
    assert result.dtype['A'] == object
    val = result[0][1]
    assert isinstance(val, Timestamp)
    assert val == df.loc[0, 'A']