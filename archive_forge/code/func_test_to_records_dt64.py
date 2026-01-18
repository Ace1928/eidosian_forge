from collections import abc
import email
from email.parser import Parser
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_records_dt64(self):
    df = DataFrame([['one', 'two', 'three'], ['four', 'five', 'six']], index=date_range('2012-01-01', '2012-01-02'))
    expected = df.index.values[0]
    result = df.to_records()['index'][0]
    assert expected == result