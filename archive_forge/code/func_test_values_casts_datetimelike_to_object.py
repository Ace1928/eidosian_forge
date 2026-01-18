import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('constructor', [date_range, period_range])
def test_values_casts_datetimelike_to_object(self, constructor):
    series = Series(constructor('2000-01-01', periods=10, freq='D'))
    expected = series.astype('object')
    df = DataFrame({'a': series, 'b': np.random.default_rng(2).standard_normal(len(series))})
    result = df.values.squeeze()
    assert (result[:, 0] == expected.values).all()
    df = DataFrame({'a': series, 'b': ['foo'] * len(series)})
    result = df.values.squeeze()
    assert (result[:, 0] == expected.values).all()