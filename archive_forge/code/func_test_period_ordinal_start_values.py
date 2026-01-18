import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.period import (
import pandas._testing as tm
@pytest.mark.parametrize('freq,expected', [('A', 0), ('M', 0), ('W', 1), ('D', 0), ('B', 0)])
def test_period_ordinal_start_values(freq, expected):
    assert period_ordinal(1970, 1, 1, 0, 0, 0, 0, 0, get_freq_code(freq)) == expected