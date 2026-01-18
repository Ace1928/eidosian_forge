from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,expected_dtype', ([np.uint64(2), int], [np.int64(-9), int], [np.float64(1.1), float], [np.bool_(True), bool], [np.datetime64('2005-02-25'), Timestamp]))
def test_to_dict_scalar_constructor_orient_dtype(self, data, expected_dtype):
    df = DataFrame({'a': data}, index=[0])
    d = df.to_dict(orient='records')
    result = type(d[0]['a'])
    assert result is expected_dtype