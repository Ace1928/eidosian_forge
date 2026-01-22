from statsmodels.compat.pandas import (
from abc import ABC, abstractmethod
import datetime as dt
from typing import Optional, Union
from collections.abc import Hashable, Sequence
import numpy as np
import pandas as pd
from scipy.linalg import qr
from statsmodels.iolib.summary import d_or_f
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import freq_to_period
def _extend_time_index(self, stop: pd.Timestamp) -> Union[pd.DatetimeIndex, pd.PeriodIndex]:
    index = self._index
    if isinstance(index, pd.PeriodIndex):
        return pd.period_range(index[0], end=stop, freq=index.freq)
    return pd.date_range(start=index[0], end=stop, freq=self._index_freq)