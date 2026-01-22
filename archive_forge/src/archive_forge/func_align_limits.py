from __future__ import annotations
import math
import typing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from ..utils import isclose_abs
from .types import DateFrequency
def align_limits(limits: TupleInt2, width: float) -> TupleFloat2:
    """
    Return limits so that breaks should be multiples of the width

    The new limits are equal or contain the original limits
    """
    low, high = limits
    l, m = divmod(low, width)
    if isclose_abs(m / width, 1):
        l += 1
    h, m = divmod(high, width)
    if not isclose_abs(m / width, 0):
        h += 1
    return (l * width, h * width)