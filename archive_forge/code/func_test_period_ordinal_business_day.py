import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.period import (
import pandas._testing as tm
@pytest.mark.parametrize('day,expected', [(3, 11415), (4, 11416), (5, 11417), (6, 11417), (7, 11417), (8, 11418)])
def test_period_ordinal_business_day(day, expected):
    args = (2013, 10, day, 0, 0, 0, 0, 0, 5000)
    assert period_ordinal(*args) == expected