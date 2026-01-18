from datetime import time
import locale
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import Series
import pandas._testing as tm
from pandas.core.tools.times import to_time
@pytest.mark.parametrize('time_string', ['14:15', '1415', pytest.param('2:15pm', marks=fails_on_non_english), pytest.param('0215pm', marks=fails_on_non_english), '14:15:00', '141500', pytest.param('2:15:00pm', marks=fails_on_non_english), pytest.param('021500pm', marks=fails_on_non_english), time(14, 15)])
def test_parsers_time(self, time_string):
    assert to_time(time_string) == time(14, 15)