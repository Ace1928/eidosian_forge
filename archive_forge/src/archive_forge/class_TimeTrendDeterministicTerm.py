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
class TimeTrendDeterministicTerm(DeterministicTerm, ABC):
    """Abstract Base Class for all Time Trend Deterministic Terms"""

    def __init__(self, constant: bool=True, order: int=0) -> None:
        self._constant = bool_like(constant, 'constant')
        self._order = required_int_like(order, 'order')

    @property
    def constant(self) -> bool:
        """Flag indicating that a constant is included"""
        return self._constant

    @property
    def order(self) -> int:
        """Order of the time trend"""
        return self._order

    @property
    def _columns(self) -> list[str]:
        columns = []
        trend_names = {1: 'trend', 2: 'trend_squared', 3: 'trend_cubed'}
        if self._constant:
            columns.append('const')
        for power in range(1, self._order + 1):
            if power in trend_names:
                columns.append(trend_names[power])
            else:
                columns.append(f'trend**{power}')
        return columns

    def _get_terms(self, locs: np.ndarray) -> np.ndarray:
        nterms = int(self._constant) + self._order
        terms = np.tile(locs, (1, nterms))
        power = np.zeros((1, nterms), dtype=int)
        power[0, int(self._constant):] = np.arange(1, self._order + 1)
        terms **= power
        return terms

    def __str__(self) -> str:
        terms = []
        if self._constant:
            terms.append('Constant')
        if self._order:
            terms.append(f'Powers 1 to {self._order + 1}')
        if not terms:
            terms = ['Empty']
        terms_str = ','.join(terms)
        return f'TimeTrend({terms_str})'