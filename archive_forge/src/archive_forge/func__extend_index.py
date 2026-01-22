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
@staticmethod
def _extend_index(index: pd.Index, steps: int, forecast_index: Optional[Sequence[Hashable]]=None) -> pd.Index:
    """Extend the forecast index"""
    if forecast_index is not None:
        forecast_index = DeterministicTerm._index_like(forecast_index)
        assert isinstance(forecast_index, pd.Index)
        if forecast_index.shape[0] != steps:
            raise ValueError(f'The number of values in forecast_index ({forecast_index.shape[0]}) must match steps ({steps}).')
        return forecast_index
    if isinstance(index, pd.PeriodIndex):
        return pd.period_range(index[-1] + 1, periods=steps, freq=index.freq)
    elif isinstance(index, pd.DatetimeIndex) and index.freq is not None:
        next_obs = pd.date_range(index[-1], freq=index.freq, periods=2)[1]
        return pd.date_range(next_obs, freq=index.freq, periods=steps)
    elif isinstance(index, pd.RangeIndex):
        assert isinstance(index, pd.RangeIndex)
        try:
            step = index.step
            start = index.stop
        except AttributeError:
            step = index[-1] - index[-2] if len(index) > 1 else 1
            start = index[-1] + step
        stop = start + step * steps
        return pd.RangeIndex(start, stop, step=step)
    elif is_int_index(index) and np.all(np.diff(index) == 1):
        idx_arr = np.arange(index[-1] + 1, index[-1] + steps + 1)
        return pd.Index(idx_arr)
    import warnings
    warnings.warn('Only PeriodIndexes, DatetimeIndexes with a frequency set, RangesIndexes, and Index with a unit increment support extending. The index is set will contain the position relative to the data length.', UserWarning, stacklevel=2)
    nobs = index.shape[0]
    return pd.RangeIndex(nobs + 1, nobs + steps + 1)