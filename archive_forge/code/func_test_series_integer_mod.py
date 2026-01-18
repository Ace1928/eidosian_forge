from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
@pytest.mark.parametrize('index', [None, range(9)])
def test_series_integer_mod(self, index):
    s1 = Series(range(1, 10))
    s2 = Series('foo', index=index)
    msg = 'not all arguments converted during string formatting|mod not'
    with pytest.raises((TypeError, NotImplementedError), match=msg):
        s2 % s1