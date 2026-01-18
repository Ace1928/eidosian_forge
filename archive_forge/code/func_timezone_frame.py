import numpy as np
import pytest
from pandas import (
@pytest.fixture
def timezone_frame():
    """
    Fixture for DataFrame of date_range Series with different time zones

    Columns are ['A', 'B', 'C']; some entries are missing

               A                         B                         C
    0 2013-01-01 2013-01-01 00:00:00-05:00 2013-01-01 00:00:00+01:00
    1 2013-01-02                       NaT                       NaT
    2 2013-01-03 2013-01-03 00:00:00-05:00 2013-01-03 00:00:00+01:00
    """
    df = DataFrame({'A': date_range('20130101', periods=3), 'B': date_range('20130101', periods=3, tz='US/Eastern'), 'C': date_range('20130101', periods=3, tz='CET')})
    df.iloc[1, 1] = NaT
    df.iloc[1, 2] = NaT
    return df