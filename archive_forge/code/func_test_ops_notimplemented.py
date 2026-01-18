from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_ops_notimplemented():

    class Other:
        pass
    other = Other()
    td = Timedelta('1 day')
    assert td.__add__(other) is NotImplemented
    assert td.__sub__(other) is NotImplemented
    assert td.__truediv__(other) is NotImplemented
    assert td.__mul__(other) is NotImplemented
    assert td.__floordiv__(other) is NotImplemented