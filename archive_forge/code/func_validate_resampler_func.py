from __future__ import annotations
from typing import (
import numpy as np
from numpy import ndarray
from pandas._libs.lib import (
from pandas.errors import UnsupportedFunctionCall
from pandas.util._validators import (
def validate_resampler_func(method: str, args, kwargs) -> None:
    """
    'args' and 'kwargs' should be empty because all of their necessary
    parameters are explicitly listed in the function signature
    """
    if len(args) + len(kwargs) > 0:
        if method in RESAMPLER_NUMPY_OPS:
            raise UnsupportedFunctionCall(f'numpy operations are not valid with resample. Use .resample(...).{method}() instead')
        raise TypeError('too many arguments passed in')