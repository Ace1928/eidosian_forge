from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_tdi_ops_attributes(self):
    rng = timedelta_range('2 days', periods=5, freq='2D', name='x')
    result = rng + 1 * rng.freq
    exp = timedelta_range('4 days', periods=5, freq='2D', name='x')
    tm.assert_index_equal(result, exp)
    assert result.freq == '2D'
    result = rng - 2 * rng.freq
    exp = timedelta_range('-2 days', periods=5, freq='2D', name='x')
    tm.assert_index_equal(result, exp)
    assert result.freq == '2D'
    result = rng * 2
    exp = timedelta_range('4 days', periods=5, freq='4D', name='x')
    tm.assert_index_equal(result, exp)
    assert result.freq == '4D'
    result = rng / 2
    exp = timedelta_range('1 days', periods=5, freq='D', name='x')
    tm.assert_index_equal(result, exp)
    assert result.freq == 'D'
    result = -rng
    exp = timedelta_range('-2 days', periods=5, freq='-2D', name='x')
    tm.assert_index_equal(result, exp)
    assert result.freq == '-2D'
    rng = timedelta_range('-2 days', periods=5, freq='D', name='x')
    result = abs(rng)
    exp = TimedeltaIndex(['2 days', '1 days', '0 days', '1 days', '2 days'], name='x')
    tm.assert_index_equal(result, exp)
    assert result.freq is None