from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_rsub_nat(self):
    td = Timedelta(10, unit='d')
    result = NaT - td
    assert result is NaT
    result = np.datetime64('NaT') - td
    assert result is NaT