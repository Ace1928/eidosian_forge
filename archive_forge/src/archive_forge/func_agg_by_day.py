import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def agg_by_day(x):
    x = x.between_time('09:00', '16:00')
    return getattr(x.rolling(5, min_periods=1), f)()