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
class FourierDeterministicTerm(DeterministicTerm, ABC):
    """Abstract Base Class for all Fourier Deterministic Terms"""

    def __init__(self, order: int) -> None:
        self._order = required_int_like(order, 'terms')

    @property
    def order(self) -> int:
        """The order of the Fourier terms included"""
        return self._order

    def _get_terms(self, locs: np.ndarray) -> np.ndarray:
        locs = 2 * np.pi * locs.astype(np.double)
        terms = np.empty((locs.shape[0], 2 * self._order))
        for i in range(self._order):
            for j, func in enumerate((np.sin, np.cos)):
                terms[:, 2 * i + j] = func((i + 1) * locs)
        return terms