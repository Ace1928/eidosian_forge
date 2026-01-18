from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_ambig(self, using_infer_string):
    dm = DataFrame(index=range(3), columns=range(3))
    coercable_series = Series([Decimal(1) for _ in range(3)], index=range(3))
    uncoercable_series = Series(['foo', 'bzr', 'baz'], index=range(3))
    dm[0] = np.ones(3)
    assert len(dm.columns) == 3
    dm[1] = coercable_series
    assert len(dm.columns) == 3
    dm[2] = uncoercable_series
    assert len(dm.columns) == 3
    if using_infer_string:
        assert dm[2].dtype == 'string'
    else:
        assert dm[2].dtype == np.object_