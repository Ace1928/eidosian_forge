from dateutil.tz import tzlocal
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_normalize_tz(self):
    rng = date_range('1/1/2000 9:30', periods=10, freq='D', tz='US/Eastern')
    result = rng.normalize()
    expected = date_range('1/1/2000', periods=10, freq='D', tz='US/Eastern')
    tm.assert_index_equal(result, expected._with_freq(None))
    assert result.is_normalized
    assert not rng.is_normalized
    rng = date_range('1/1/2000 9:30', periods=10, freq='D', tz='UTC')
    result = rng.normalize()
    expected = date_range('1/1/2000', periods=10, freq='D', tz='UTC')
    tm.assert_index_equal(result, expected)
    assert result.is_normalized
    assert not rng.is_normalized
    rng = date_range('1/1/2000 9:30', periods=10, freq='D', tz=tzlocal())
    result = rng.normalize()
    expected = date_range('1/1/2000', periods=10, freq='D', tz=tzlocal())
    tm.assert_index_equal(result, expected._with_freq(None))
    assert result.is_normalized
    assert not rng.is_normalized