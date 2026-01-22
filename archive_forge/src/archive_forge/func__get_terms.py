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
def _get_terms(self, index: Union[pd.DatetimeIndex, pd.PeriodIndex]) -> np.ndarray:
    if self._period == 'D':
        locs = self._daily_to_loc(index)
    elif self._period == 'W':
        locs = self._weekly_to_loc(index)
    elif self._period in ('Q', 'QE'):
        locs = self._quarterly_to_loc(index)
    else:
        locs = self._annual_to_loc(index)
    full_cycle = self._supported[self._period][self._freq_str]
    terms = np.zeros((locs.shape[0], full_cycle))
    terms[np.arange(locs.shape[0]), locs] = 1
    return terms