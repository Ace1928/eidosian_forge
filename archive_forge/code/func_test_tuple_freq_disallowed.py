from datetime import (
import re
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_tuple_freq_disallowed(self):
    with pytest.raises(TypeError, match='pass as a string instead'):
        Period('1982', freq=('Min', 1))
    with pytest.raises(TypeError, match='pass as a string instead'):
        Period('2006-12-31', ('w', 1))