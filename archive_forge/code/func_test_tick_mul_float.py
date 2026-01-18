from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import delta_to_tick
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import INT_NEG_999_TO_POS_999
from pandas.tests.tseries.offsets.common import assert_offset_equal
from pandas.tseries import offsets
from pandas.tseries.offsets import (
def test_tick_mul_float():
    off = Micro(2)
    result = off * 1.5
    expected = Micro(3)
    assert result == expected
    assert isinstance(result, Micro)
    result = off * 1.25
    expected = Nano(2500)
    assert result == expected
    assert isinstance(result, Nano)