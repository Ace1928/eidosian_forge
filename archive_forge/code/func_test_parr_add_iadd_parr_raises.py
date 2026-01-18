import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
def test_parr_add_iadd_parr_raises(self, box_with_array):
    rng = period_range('1/1/2000', freq='D', periods=5)
    other = period_range('1/6/2000', freq='D', periods=5)
    rng = tm.box_expected(rng, box_with_array)
    msg = 'unsupported operand type\\(s\\) for \\+: .* and .*'
    with pytest.raises(TypeError, match=msg):
        rng + other
    with pytest.raises(TypeError, match=msg):
        rng += other