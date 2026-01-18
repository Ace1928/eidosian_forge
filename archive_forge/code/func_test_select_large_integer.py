import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import Term
def test_select_large_integer(tmp_path):
    path = tmp_path / 'large_int.h5'
    df = DataFrame(zip(['a', 'b', 'c', 'd'], [-9223372036854775801, -9223372036854775802, -9223372036854775803, 123]), columns=['x', 'y'])
    result = None
    with HDFStore(path) as s:
        s.append('data', df, data_columns=True, index=False)
        result = s.select('data', where='y==-9223372036854775801').get('y').get(0)
    expected = df['y'][0]
    assert expected == result