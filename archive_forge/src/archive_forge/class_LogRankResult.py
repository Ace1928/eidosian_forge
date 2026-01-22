from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import warnings
import numpy as np
from scipy import special, interpolate, stats
from scipy.stats._censored_data import CensoredData
from scipy.stats._common import ConfidenceInterval
@dataclass
class LogRankResult:
    """Result object returned by `scipy.stats.logrank`.

    Attributes
    ----------
    statistic : float ndarray
        The computed statistic (defined below). Its magnitude is the
        square root of the magnitude returned by most other logrank test
        implementations.
    pvalue : float ndarray
        The computed p-value of the test.
    """
    statistic: np.ndarray
    pvalue: np.ndarray