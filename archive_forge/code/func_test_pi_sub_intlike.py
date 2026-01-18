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
@pytest.mark.parametrize('five', [5, np.array(5, dtype=np.int64)])
def test_pi_sub_intlike(self, five):
    rng = period_range('2007-01', periods=50)
    result = rng - five
    exp = rng + -five
    tm.assert_index_equal(result, exp)