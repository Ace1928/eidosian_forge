from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('op', [operator.add, ops.radd])
def test_td_add_td(self, op):
    td = Timedelta(10, unit='d')
    result = op(td, Timedelta(days=10))
    assert isinstance(result, Timedelta)
    assert result == Timedelta(days=20)