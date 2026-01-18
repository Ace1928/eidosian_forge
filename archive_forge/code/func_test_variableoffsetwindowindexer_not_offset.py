import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import (
from pandas.core.indexers.objects import (
from pandas.tseries.offsets import BusinessDay
def test_variableoffsetwindowindexer_not_offset():
    idx = date_range('2020', periods=10)
    with pytest.raises(ValueError, match='offset must be a DateOffset-like object.'):
        VariableOffsetWindowIndexer(index=idx, offset='foo')