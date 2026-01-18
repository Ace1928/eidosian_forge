import datetime as dt
from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nunique_with_object():
    data = DataFrame([[100, 1, 'Alice'], [200, 2, 'Bob'], [300, 3, 'Charlie'], [-400, 4, 'Dan'], [500, 5, 'Edith']], columns=['amount', 'id', 'name'])
    result = data.groupby(['id', 'amount'])['name'].nunique()
    index = MultiIndex.from_arrays([data.id, data.amount])
    expected = Series([1] * 5, name='name', index=index)
    tm.assert_series_equal(result, expected)