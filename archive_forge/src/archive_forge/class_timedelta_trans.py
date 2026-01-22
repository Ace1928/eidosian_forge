from __future__ import annotations
import sys
import typing
from abc import ABC, abstractmethod
from datetime import MAXYEAR, MINYEAR, datetime, timedelta
from types import MethodType
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from ._core.dates import datetime_to_num, num_to_datetime
from .breaks import (
from .labels import (
from .utils import identity
class timedelta_trans(trans):
    """
    Timedelta Transformation
    """
    domain = (timedelta.min, timedelta.max)
    breaks_ = staticmethod(breaks_timedelta())
    format = staticmethod(label_timedelta())

    @staticmethod
    def transform(x: NDArrayTimedelta | Sequence[timedelta]) -> NDArrayFloat:
        """
        Transform from Timeddelta to numerical format
        """
        return np.array([_x.total_seconds() * 10 ** 6 for _x in x])

    @staticmethod
    def inverse(x: FloatArrayLike) -> NDArrayTimedelta:
        """
        Transform to Timedelta from numerical format
        """
        return np.array([timedelta(microseconds=i) for i in x])