from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,dtype', (([True, True, False], bool), [[datetime(2018, 1, 1), datetime(2019, 2, 2), datetime(2020, 3, 3)], Timestamp], [[1.0, 2.0, 3.0], float], [[1, 2, 3], int], [['X', 'Y', 'Z'], str]))
def test_to_dict_orient_dtype(self, data, dtype):
    df = DataFrame({'a': data})
    d = df.to_dict(orient='records')
    assert all((type(record['a']) is dtype for record in d))