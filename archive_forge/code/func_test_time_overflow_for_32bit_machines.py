import datetime as dt
from datetime import date
import re
import numpy as np
import pytest
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_time_overflow_for_32bit_machines(self):
    periods = np_long(1000)
    idx1 = date_range(start='2000', periods=periods, freq='s')
    assert len(idx1) == periods
    idx2 = date_range(end='2000', periods=periods, freq='s')
    assert len(idx2) == periods