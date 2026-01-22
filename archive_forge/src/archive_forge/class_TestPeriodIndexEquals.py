from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestPeriodIndexEquals(EqualsTests):

    @pytest.fixture
    def index(self):
        return period_range('2013-01-01', periods=5, freq='D')

    @pytest.mark.parametrize('freq', ['D', 'M'])
    def test_equals2(self, freq):
        idx = PeriodIndex(['2011-01-01', '2011-01-02', 'NaT'], freq=freq)
        assert idx.equals(idx)
        assert idx.equals(idx.copy())
        assert idx.equals(idx.astype(object))
        assert idx.astype(object).equals(idx)
        assert idx.astype(object).equals(idx.astype(object))
        assert not idx.equals(list(idx))
        assert not idx.equals(pd.Series(idx))
        idx2 = PeriodIndex(['2011-01-01', '2011-01-02', 'NaT'], freq='h')
        assert not idx.equals(idx2)
        assert not idx.equals(idx2.copy())
        assert not idx.equals(idx2.astype(object))
        assert not idx.astype(object).equals(idx2)
        assert not idx.equals(list(idx2))
        assert not idx.equals(pd.Series(idx2))
        idx3 = PeriodIndex._simple_new(idx._values._simple_new(idx._values.asi8, dtype=pd.PeriodDtype('h')))
        tm.assert_numpy_array_equal(idx.asi8, idx3.asi8)
        assert not idx.equals(idx3)
        assert not idx.equals(idx3.copy())
        assert not idx.equals(idx3.astype(object))
        assert not idx.astype(object).equals(idx3)
        assert not idx.equals(list(idx3))
        assert not idx.equals(pd.Series(idx3))