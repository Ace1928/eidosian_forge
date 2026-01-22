from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
class minor_breaks:
    """
    Compute minor breaks

    This is the naive method. It does not take into account
    the transformation.

    Parameters
    ----------
    n : int
        Number of minor breaks between the major
        breaks.

    Examples
    --------
    >>> major = [1, 2, 3, 4]
    >>> limits = [0, 5]
    >>> minor_breaks()(major, limits)
    array([0.5, 1.5, 2.5, 3.5, 4.5])
    >>> minor_breaks()([1, 2], (1, 2))
    array([1.5])

    More than 1 minor break.

    >>> minor_breaks(3)([1, 2], (1, 2))
    array([1.25, 1.5 , 1.75])
    >>> minor_breaks()([1, 2], (1, 2), 3)
    array([1.25, 1.5 , 1.75])
    """

    def __init__(self, n: int=1):
        self.n = n

    def __call__(self, major: FloatArrayLike, limits: Optional[TupleFloat2]=None, n: Optional[int]=None) -> NDArrayFloat:
        """
        Minor breaks

        Parameters
        ----------
        major : array_like
            Major breaks
        limits : array_like | None
            Limits of the scale. If *array_like*, must be
            of size 2. If **None**, then the minimum and
            maximum of the major breaks are used.
        n : int
            Number of minor breaks between the major
            breaks. If **None**, then *self.n* is used.

        Returns
        -------
        out : array_like
            Minor beraks
        """
        if len(major) < 2:
            return np.array([])
        if limits is None:
            low, high = min_max(major)
        else:
            low, high = min_max(limits)
        if n is None:
            n = self.n
        diff = np.diff(major)
        step = diff[0]
        if len(diff) > 1 and all(diff == step):
            major = np.hstack([major[0] - step, major, major[-1] + step])
        mbreaks = []
        factors = np.arange(1, n + 1)
        for lhs, rhs in zip(major[:-1], major[1:]):
            sep = (rhs - lhs) / (n + 1)
            mbreaks.append(lhs + factors * sep)
        minor = np.hstack(mbreaks)
        minor = minor.compress((low <= minor) & (minor <= high))
        return minor