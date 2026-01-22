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
def _annual_to_loc(self, index: Union[pd.DatetimeIndex, pd.PeriodIndex]) -> np.ndarray:
    if self._freq.freqstr in ('M', 'ME', 'MS'):
        return index.month - 1
    else:
        return index.quarter - 1