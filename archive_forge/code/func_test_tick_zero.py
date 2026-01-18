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
@pytest.mark.parametrize('cls1', tick_classes)
@pytest.mark.parametrize('cls2', tick_classes)
def test_tick_zero(cls1, cls2):
    assert cls1(0) == cls2(0)
    assert cls1(0) + cls2(0) == cls1(0)
    if cls1 is not Nano:
        assert cls1(2) + cls2(0) == cls1(2)
    if cls1 is Nano:
        assert cls1(2) + Nano(0) == cls1(2)