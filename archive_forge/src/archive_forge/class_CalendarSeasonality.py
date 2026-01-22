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
class CalendarSeasonality(CalendarDeterministicTerm):
    """
    Seasonal dummy deterministic terms based on calendar time

    Parameters
    ----------
    freq : str
        The frequency of the seasonal effect.
    period : str
        The pandas frequency string describing the full period.

    See Also
    --------
    DeterministicProcess
    CalendarTimeTrend
    CalendarFourier
    Seasonality

    Examples
    --------
    Here we simulate irregularly spaced data (in time) and hourly seasonal
    dummies for the data.

    >>> import numpy as np
    >>> import pandas as pd
    >>> base = pd.Timestamp("2020-1-1")
    >>> gen = np.random.default_rng()
    >>> gaps = np.cumsum(gen.integers(0, 1800, size=1000))
    >>> times = [base + pd.Timedelta(gap, unit="s") for gap in gaps]
    >>> index = pd.DatetimeIndex(pd.to_datetime(times))

    >>> from statsmodels.tsa.deterministic import CalendarSeasonality
    >>> cal_seas_gen = CalendarSeasonality("H", "D")
    >>> cal_seas_gen.in_sample(index)
    """
    _is_dummy = True
    if PD_LT_2_2_0:
        _supported = {'W': {'B': 5, 'D': 7, 'h': 24 * 7, 'H': 24 * 7}, 'D': {'h': 24, 'H': 24}, 'Q': {'MS': 3, 'M': 3}, 'A': {'MS': 12, 'M': 12}, 'Y': {'MS': 12, 'Q': 4, 'M': 12}}
    else:
        _supported = {'W': {'B': 5, 'D': 7, 'h': 24 * 7}, 'D': {'h': 24}, 'Q': {'MS': 3, 'ME': 3}, 'A': {'MS': 12, 'ME': 12, 'QE': 4}, 'Y': {'MS': 12, 'ME': 12, 'QE': 4}, 'QE': {'ME': 3}, 'YE': {'ME': 12, 'QE': 4}}

    def __init__(self, freq: str, period: str) -> None:
        freq_options: set[str] = set()
        freq_options.update(*[list(val.keys()) for val in self._supported.values()])
        period_options = tuple(self._supported.keys())
        freq = string_like(freq, 'freq', options=tuple(freq_options), lower=False)
        period = string_like(period, 'period', options=period_options, lower=False)
        if freq not in self._supported[period]:
            raise ValueError(f'The combination of freq={freq} and period={period} is not supported.')
        super().__init__(freq)
        self._period = period
        self._freq_str = self._freq.freqstr.split('-')[0]

    @property
    def freq(self) -> str:
        """The frequency of the deterministic terms"""
        return self._freq.freqstr

    @property
    def period(self) -> str:
        """The full period"""
        return self._period

    def _weekly_to_loc(self, index: Union[pd.DatetimeIndex, pd.PeriodIndex]) -> np.ndarray:
        if self._freq.freqstr in ('h', 'H'):
            return index.hour + 24 * index.dayofweek
        elif self._freq.freqstr == 'D':
            return index.dayofweek
        else:
            bdays = pd.bdate_range('2000-1-1', periods=10).dayofweek.unique()
            loc = index.dayofweek
            if not loc.isin(bdays).all():
                raise ValueError('freq is B but index contains days that are not business days.')
            return loc

    def _daily_to_loc(self, index: Union[pd.DatetimeIndex, pd.PeriodIndex]) -> np.ndarray:
        return index.hour

    def _quarterly_to_loc(self, index: Union[pd.DatetimeIndex, pd.PeriodIndex]) -> np.ndarray:
        return (index.month - 1) % 3

    def _annual_to_loc(self, index: Union[pd.DatetimeIndex, pd.PeriodIndex]) -> np.ndarray:
        if self._freq.freqstr in ('M', 'ME', 'MS'):
            return index.month - 1
        else:
            return index.quarter - 1

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

    @property
    def _columns(self) -> list[str]:
        columns = []
        count = self._supported[self._period][self._freq_str]
        for i in range(count):
            columns.append(f's({self._freq_str}={i + 1}, period={self._period})')
        return columns

    @Appender(DeterministicTerm.in_sample.__doc__)
    def in_sample(self, index: Union[Sequence[Hashable], pd.Index]) -> pd.DataFrame:
        index = self._index_like(index)
        index = self._check_index_type(index)
        terms = self._get_terms(index)
        return pd.DataFrame(terms, index=index, columns=self._columns)

    @Appender(DeterministicTerm.out_of_sample.__doc__)
    def out_of_sample(self, steps: int, index: Union[Sequence[Hashable], pd.Index], forecast_index: Optional[Sequence[Hashable]]=None) -> pd.DataFrame:
        index = self._index_like(index)
        fcast_index = self._extend_index(index, steps, forecast_index)
        self._check_index_type(fcast_index)
        assert isinstance(fcast_index, (pd.DatetimeIndex, pd.PeriodIndex))
        terms = self._get_terms(fcast_index)
        return pd.DataFrame(terms, index=fcast_index, columns=self._columns)

    @property
    def _eq_attr(self) -> tuple[Hashable, ...]:
        return (self._period, self._freq_str)

    def __str__(self) -> str:
        return f'Seasonal(freq={self._freq_str})'