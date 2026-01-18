from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_rsub_offset(self):
    result = offsets.Hour(1) - Timedelta(10, unit='d')
    assert isinstance(result, Timedelta)
    assert result == Timedelta(-239, unit='h')