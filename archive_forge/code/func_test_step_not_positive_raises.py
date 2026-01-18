from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_step_not_positive_raises():
    with pytest.raises(ValueError, match='step must be >= 0'):
        DataFrame(range(2)).rolling(1, step=-1)