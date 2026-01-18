from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_rmod_timedelta64(self):
    td = Timedelta(minutes=3)
    result = np.timedelta64(5, 'm') % td
    assert isinstance(result, Timedelta)
    assert result == Timedelta(minutes=2)