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
class identity_trans(trans):
    """
    Identity Transformation

    Examples
    --------
    The default trans returns one minor break between every pair
    of major break

    >>> major = [0, 1, 2]
    >>> t = identity_trans()
    >>> t.minor_breaks(major)
    array([0.5, 1.5])

    Create a trans that returns 4 minor breaks

    >>> t = identity_trans(minor_breaks=minor_breaks(4))
    >>> t.minor_breaks(major)
    array([0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8])
    """
    transform_is_linear = True
    transform = staticmethod(identity)
    inverse = staticmethod(identity)