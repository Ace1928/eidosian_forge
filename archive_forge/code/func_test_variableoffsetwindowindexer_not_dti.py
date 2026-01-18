import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import (
from pandas.core.indexers.objects import (
from pandas.tseries.offsets import BusinessDay
def test_variableoffsetwindowindexer_not_dti():
    with pytest.raises(ValueError, match='index must be a DatetimeIndex.'):
        VariableOffsetWindowIndexer(index='foo', offset=BusinessDay(1))