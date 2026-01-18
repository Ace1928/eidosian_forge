from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
def test_period_can_hold_element(self, element):
    pi = period_range('2016', periods=3, freq='Y')
    elem = element(pi)
    self.check_series_setitem(elem, pi, True)
    pi2 = pi.asfreq('D')[:-1]
    elem = element(pi2)
    with tm.assert_produces_warning(FutureWarning):
        self.check_series_setitem(elem, pi, False)
    dti = pi.to_timestamp('s')[:-1]
    elem = element(dti)
    with tm.assert_produces_warning(FutureWarning):
        self.check_series_setitem(elem, pi, False)