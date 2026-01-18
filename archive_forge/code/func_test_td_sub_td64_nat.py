from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_sub_td64_nat(self):
    td = Timedelta(10, unit='d')
    td_nat = np.timedelta64('NaT')
    result = td - td_nat
    assert result is NaT
    result = td_nat - td
    assert result is NaT