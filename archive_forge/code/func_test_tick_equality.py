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
@pytest.mark.arm_slow
@pytest.mark.parametrize('cls', tick_classes)
@example(n=2, m=3)
@given(n=INT_NEG_999_TO_POS_999, m=INT_NEG_999_TO_POS_999)
def test_tick_equality(cls, n, m):
    assume(m != n)
    left = cls(n)
    right = cls(m)
    assert left != right
    right = cls(n)
    assert left == right
    assert not left != right
    if n != 0:
        assert cls(n) != cls(-n)