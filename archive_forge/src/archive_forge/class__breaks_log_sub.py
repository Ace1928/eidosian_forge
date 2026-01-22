from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
class _breaks_log_sub:
    """
    Breaks for log transformed scales

    Calculate breaks that do not fall on integer powers of
    the base.

    Parameters
    ----------
    n : int
        Desired number of breaks
    base : int | float
        Base of logarithm

    Notes
    -----
    Credit: Thierry Onkelinx (thierry.onkelinx@inbo.be) for the original
    algorithm in the r-scales package.
    """

    def __init__(self, n: int=5, base: float=10):
        self.n = n
        self.base = base

    def __call__(self, limits: TupleFloat2) -> NDArrayFloat:
        base = self.base
        n = self.n
        rng = log(limits, base)
        _min = int(np.floor(rng[0]))
        _max = int(np.ceil(rng[1]))
        steps = [1]
        if float(base) ** _max > sys.maxsize:
            base = float(base)

        def delta(x):
            """
            Calculates the smallest distance in the log scale between the
            currently selected breaks and a new candidate 'x'
            """
            arr = np.sort(np.hstack([x, steps, base]))
            log_arr = log(arr, base)
            return np.min(np.diff(log_arr))
        if self.base == 2:
            return np.array([base ** i for i in range(_min, _max + 1)])
        candidate = np.arange(base + 1)
        candidate = np.compress((candidate > 1) & (candidate < base), candidate)
        while len(candidate):
            best = np.argmax([delta(x) for x in candidate])
            steps.append(candidate[best])
            candidate = np.delete(candidate, best)
            _factors = [base ** i for i in range(_min, _max + 1)]
            breaks = np.array([f * s for f, s in product(_factors, steps)])
            relevant_breaks = (limits[0] <= breaks) & (breaks <= limits[1])
            if np.sum(relevant_breaks) >= n - 2:
                breaks = np.sort(breaks)
                lower_end = np.max([np.min(np.where(limits[0] <= breaks)) - 1, 0])
                upper_end = np.min([np.max(np.where(breaks <= limits[1])) + 1, len(breaks)])
                return breaks[lower_end:upper_end + 1]
        else:
            return breaks_extended(n=n)(limits)