from __future__ import annotations
import warnings
from typing import Callable
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.core.utils import is_duck_array, module_available
from xarray.namedarray import pycompat
from xarray.core.options import OPTIONS
def _ensure_bool_is_ndarray(result, *args):
    if isinstance(result, bool):
        shape = np.broadcast(*args).shape
        constructor = np.ones if result else np.zeros
        result = constructor(shape, dtype=bool)
    return result