from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
def test_pickle_compat_construction(self):
    msg = '|'.join(['Index\\(\\.\\.\\.\\) must be called with a collection of some kind, None was passed', 'DatetimeIndex\\(\\) must be called with a collection of some kind, None was passed', 'TimedeltaIndex\\(\\) must be called with a collection of some kind, None was passed', "__new__\\(\\) missing 1 required positional argument: 'data'", '__new__\\(\\) takes at least 2 arguments \\(1 given\\)'])
    with pytest.raises(TypeError, match=msg):
        self._index_cls()