import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('freq', ['1D', offsets.Day(2), '2ms'])
def test_valid_window(self, freq, regular):
    regular.rolling(window=freq)