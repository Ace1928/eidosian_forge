from __future__ import annotations
from collections import defaultdict
from typing import cast
import numpy as np
from pandas.core.dtypes.generic import (
from pandas.core.indexes.api import MultiIndex
def zsqrt(x):
    with np.errstate(all='ignore'):
        result = np.sqrt(x)
        mask = x < 0
    if isinstance(x, ABCDataFrame):
        if mask._values.any():
            result[mask] = 0
    elif mask.any():
        result[mask] = 0
    return result