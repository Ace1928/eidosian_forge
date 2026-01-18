import datetime
import importlib
import inspect
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import six  # type: ignore
from triad.utils.assertion import assert_or_throw
def to_timedelta(obj: Any) -> datetime.timedelta:
    """Convert an object to python datetime.

    If the object is a string, `min` or `-inf` will return `timedelta.min`,
    `max` or `inf` will return `timedelta.max`; if the object is a number,
    the number will be used as the seconds argument; Otherwise it will use
    `pandas.to_timedelta` to parse the object.

    :param obj: object
    :raises TypeError: if failed to convert
    :return: timedelta value
    """
    if obj is None:
        raise TypeError("None can't convert to timedelta")
    if isinstance(obj, datetime.timedelta):
        return obj
    if np.isreal(obj):
        return datetime.timedelta(seconds=float(obj))
    try:
        return pd.to_timedelta(obj).to_pytimedelta()
    except Exception as e:
        if isinstance(obj, str):
            obj = obj.lower()
            if obj in ['min', '-inf']:
                return datetime.timedelta.min
            elif obj in ['max', 'inf']:
                return datetime.timedelta.max
        raise TypeError(f"{type(obj)} {obj} can't convert to timedelta", e)