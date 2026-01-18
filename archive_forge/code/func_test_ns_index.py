import datetime as dt
from datetime import date
import re
import numpy as np
import pytest
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_ns_index(self):
    nsamples = 400
    ns = int(1000000000.0 / 24414)
    dtstart = np.datetime64('2012-09-20T00:00:00')
    dt = dtstart + np.arange(nsamples) * np.timedelta64(ns, 'ns')
    freq = ns * offsets.Nano()
    index = DatetimeIndex(dt, freq=freq, name='time')
    self.assert_index_parameters(index)
    new_index = date_range(start=index[0], end=index[-1], freq=index.freq)
    self.assert_index_parameters(new_index)