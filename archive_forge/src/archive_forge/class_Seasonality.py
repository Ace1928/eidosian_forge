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
class Seasonality(DeterministicTerm):
    """
    Seasonal dummy deterministic terms

    Parameters
    ----------
    period : int
        The length of a full cycle. Must be >= 2.
    initial_period : int
        The seasonal index of the first observation. 1-indexed so must
        be in {1, 2, ..., period}.

    See Also
    --------
    DeterministicProcess
    TimeTrend
    Fourier
    CalendarSeasonality

    Examples
    --------
    Solar data has an 11-year cycle

    >>> from statsmodels.datasets import sunspots
    >>> from statsmodels.tsa.deterministic import Seasonality
    >>> data = sunspots.load_pandas().data
    >>> seas_gen = Seasonality(11)
    >>> seas_gen.in_sample(data.index)

    To start at a season other than 1

    >>> seas_gen = Seasonality(11, initial_period=4)
    >>> seas_gen.in_sample(data.index)
    """
    _is_dummy = True

    def __init__(self, period: int, initial_period: int=1) -> None:
        self._period = required_int_like(period, 'period')
        self._initial_period = required_int_like(initial_period, 'initial_period')
        if period < 2:
            raise ValueError('period must be >= 2')
        if not 1 <= self._initial_period <= period:
            raise ValueError('initial_period must be in {1, 2, ..., period}')

    @property
    def period(self) -> int:
        """The period of the seasonality"""
        return self._period

    @property
    def initial_period(self) -> int:
        """The seasonal index of the first observation"""
        return self._initial_period

    @classmethod
    def from_index(cls, index: Union[Sequence[Hashable], pd.DatetimeIndex, pd.PeriodIndex]) -> 'Seasonality':
        """
        Construct a seasonality directly from an index using its frequency.

        Parameters
        ----------
        index : {DatetimeIndex, PeriodIndex}
            An index with its frequency (`freq`) set.

        Returns
        -------
        Seasonality
            The initialized Seasonality instance.
        """
        index = cls._index_like(index)
        if isinstance(index, pd.PeriodIndex):
            freq = index.freq
        elif isinstance(index, pd.DatetimeIndex):
            freq = index.freq if index.freq else index.inferred_freq
        else:
            raise TypeError('index must be a DatetimeIndex or PeriodIndex')
        if freq is None:
            raise ValueError('index must have a freq or inferred_freq set')
        period = freq_to_period(freq)
        return cls(period=period)

    @property
    def _eq_attr(self) -> tuple[Hashable, ...]:
        return (self._period, self._initial_period)

    def __str__(self) -> str:
        return f'Seasonality(period={self._period})'

    @property
    def _columns(self) -> list[str]:
        period = self._period
        columns = []
        for i in range(1, period + 1):
            columns.append(f's({i},{period})')
        return columns

    @Appender(DeterministicTerm.in_sample.__doc__)
    def in_sample(self, index: Union[Sequence[Hashable], pd.Index]) -> pd.DataFrame:
        index = self._index_like(index)
        nobs = index.shape[0]
        period = self._period
        term = np.zeros((nobs, period))
        offset = self._initial_period - 1
        for i in range(period):
            col = (i + offset) % period
            term[i::period, col] = 1
        return pd.DataFrame(term, columns=self._columns, index=index)

    @Appender(DeterministicTerm.out_of_sample.__doc__)
    def out_of_sample(self, steps: int, index: Union[Sequence[Hashable], pd.Index], forecast_index: Optional[Sequence[Hashable]]=None) -> pd.DataFrame:
        index = self._index_like(index)
        fcast_index = self._extend_index(index, steps, forecast_index)
        nobs = index.shape[0]
        period = self._period
        term = np.zeros((steps, period))
        offset = self._initial_period - 1
        for i in range(period):
            col_loc = (nobs + offset + i) % period
            term[i::period, col_loc] = 1
        return pd.DataFrame(term, columns=self._columns, index=fcast_index)