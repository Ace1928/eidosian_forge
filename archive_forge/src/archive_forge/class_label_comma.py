from __future__ import annotations
import re
import typing
from bisect import bisect_right
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import numpy as np
from .breaks import timedelta_helper
from .utils import (
@dataclass
class label_comma(label_currency):
    """
    Labels of numbers with commas as separators

    Parameters
    ----------
    precision : int
        Number of digits after the decimal point.

    Examples
    --------
    >>> label_comma()([1000, 2, 33000, 400])
    ['1,000', '2', '33,000', '400']
    """
    prefix: str = ''
    precision: int = 0
    big_mark: str = ','