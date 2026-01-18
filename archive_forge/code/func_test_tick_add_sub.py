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
@pytest.mark.parametrize('cls', tick_classes)
@example(n=2, m=3)
@example(n=800, m=300)
@example(n=1000, m=5)
@given(n=INT_NEG_999_TO_POS_999, m=INT_NEG_999_TO_POS_999)
def test_tick_add_sub(cls, n, m):
    left = cls(n)
    right = cls(m)
    expected = cls(n + m)
    assert left + right == expected
    expected = cls(n - m)
    assert left - right == expected