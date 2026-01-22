from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
class breaks_timedelta:
    """
    Timedelta breaks

    Returns
    -------
    out : callable ``f(limits)``
        A function that takes a sequence of two
        :class:`datetime.timedelta` values and returns
        a sequence of break points.

    Examples
    --------
    >>> from datetime import timedelta
    >>> breaks = breaks_timedelta()
    >>> x = [timedelta(days=i*365) for i in range(25)]
    >>> limits = min(x), max(x)
    >>> major = breaks(limits)
    >>> [val.total_seconds()/(365*24*60*60)for val in major]
    [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    """
    _calculate_breaks: Callable[[TupleFloat2], NDArrayFloat]

    def __init__(self, n: int=5, Q: Sequence[float]=(1, 2, 5, 10)):
        self._calculate_breaks = breaks_extended(n=n, Q=Q)

    def __call__(self, limits: tuple[Timedelta, Timedelta]) -> NDArrayTimedelta:
        """
        Compute breaks

        Parameters
        ----------
        limits : tuple
            Minimum and maximum :class:`datetime.timedelta` values.

        Returns
        -------
        out : array_like
            Sequence of break points.
        """
        if any((pd.isna(x) for x in limits)):
            return np.array([])
        helper = timedelta_helper(limits)
        scaled_limits = helper.scaled_limits()
        scaled_breaks = self._calculate_breaks(scaled_limits)
        breaks = helper.numeric_to_timedelta(scaled_breaks)
        return breaks