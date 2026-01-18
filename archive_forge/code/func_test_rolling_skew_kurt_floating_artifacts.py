from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_skew_kurt_floating_artifacts():
    sr = Series([1 / 3, 4, 0, 0, 0, 0, 0])
    r = sr.rolling(4)
    result = r.skew()
    assert (result[-2:] == 0).all()
    result = r.kurt()
    assert (result[-2:] == -3).all()