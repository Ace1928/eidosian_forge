from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
class breaks_date:
    """
    Regularly spaced dates

    Parameters
    ----------
    n :
        Desired number of breaks.
    width : str | None
        An interval specification. Must be one of
        [second, minute, hour, day, week, month, year]
        If ``None``, the interval automatic.

    Examples
    --------
    >>> from datetime import datetime
    >>> limits = (datetime(2010, 1, 1), datetime(2026, 1, 1))

    Default breaks will be regularly spaced but the spacing
    is automatically determined

    >>> breaks = breaks_date(9)
    >>> [d.year for d in breaks(limits)]
    [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026]

    Breaks at 4 year intervals

    >>> breaks = breaks_date('4 year')
    >>> [d.year for d in breaks(limits)]
    [2010, 2014, 2018, 2022, 2026]
    """
    n: int
    width: Optional[int] = None
    units: Optional[DatetimeBreaksUnits] = None

    def __init__(self, n: int=5, width: Optional[str]=None):
        if isinstance(n, str):
            width = n
        self.n = n
        if width:
            _w, units = width.strip().lower().split()
            self.width = int(_w)
            self.units = units.rstrip('s')

    def __call__(self, limits: TupleT2[datetime]) -> Sequence[datetime]:
        """
        Compute breaks

        Parameters
        ----------
        limits : tuple
            Minimum and maximum :class:`datetime.datetime` values.

        Returns
        -------
        out : array_like
            Sequence of break points.
        """
        if any((pd.isna(x) for x in limits)):
            return []
        if isinstance(limits[0], np.datetime64) and isinstance(limits[1], np.datetime64):
            limits = (limits[0].astype(object), limits[1].astype(object))
        if self.units and self.width:
            return calculate_date_breaks_byunits(limits, self.units, self.width)
        else:
            return calculate_date_breaks_auto(limits, self.n)