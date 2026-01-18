from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_map_box_timedelta():
    ser = Series(timedelta_range('1 day 1 s', periods=5, freq='h'))

    def f(x):
        return x.total_seconds()
    ser.map(f)