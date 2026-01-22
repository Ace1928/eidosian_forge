import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestPeriodRangeKeywords:

    def test_required_arguments(self):
        msg = 'Of the three parameters: start, end, and periods, exactly two must be specified'
        with pytest.raises(ValueError, match=msg):
            period_range('2011-1-1', '2012-1-1', 'B')

    def test_required_arguments2(self):
        start = Period('02-Apr-2005', 'D')
        msg = 'Of the three parameters: start, end, and periods, exactly two must be specified'
        with pytest.raises(ValueError, match=msg):
            period_range(start=start)

    def test_required_arguments3(self):
        msg = 'Of the three parameters: start, end, and periods, exactly two must be specified'
        with pytest.raises(ValueError, match=msg):
            period_range(start='2017Q1')
        with pytest.raises(ValueError, match=msg):
            period_range(end='2017Q1')
        with pytest.raises(ValueError, match=msg):
            period_range(periods=5)
        with pytest.raises(ValueError, match=msg):
            period_range()

    def test_required_arguments_too_many(self):
        msg = 'Of the three parameters: start, end, and periods, exactly two must be specified'
        with pytest.raises(ValueError, match=msg):
            period_range(start='2017Q1', end='2018Q1', periods=8, freq='Q')

    def test_start_end_non_nat(self):
        msg = 'start and end must not be NaT'
        with pytest.raises(ValueError, match=msg):
            period_range(start=NaT, end='2018Q1')
        with pytest.raises(ValueError, match=msg):
            period_range(start=NaT, end='2018Q1', freq='Q')
        with pytest.raises(ValueError, match=msg):
            period_range(start='2017Q1', end=NaT)
        with pytest.raises(ValueError, match=msg):
            period_range(start='2017Q1', end=NaT, freq='Q')

    def test_periods_requires_integer(self):
        msg = 'periods must be a number, got foo'
        with pytest.raises(TypeError, match=msg):
            period_range(start='2017Q1', periods='foo')