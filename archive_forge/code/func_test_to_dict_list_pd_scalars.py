from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val', [Timestamp(2020, 1, 1), Timedelta(1), Period('2020'), Interval(1, 2)])
def test_to_dict_list_pd_scalars(val):
    df = DataFrame({'a': [val]})
    result = df.to_dict(orient='list')
    expected = {'a': [val]}
    assert result == expected