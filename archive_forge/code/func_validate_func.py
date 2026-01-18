from __future__ import annotations
from typing import (
import numpy as np
from numpy import ndarray
from pandas._libs.lib import (
from pandas.errors import UnsupportedFunctionCall
from pandas.util._validators import (
def validate_func(fname, args, kwargs) -> None:
    if fname not in _validation_funcs:
        return validate_stat_func(args, kwargs, fname=fname)
    validation_func = _validation_funcs[fname]
    return validation_func(args, kwargs)