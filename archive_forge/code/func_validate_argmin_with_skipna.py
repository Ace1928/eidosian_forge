from __future__ import annotations
from typing import (
import numpy as np
from numpy import ndarray
from pandas._libs.lib import (
from pandas.errors import UnsupportedFunctionCall
from pandas.util._validators import (
def validate_argmin_with_skipna(skipna: bool | ndarray | None, args, kwargs) -> bool:
    """
    If 'Series.argmin' is called via the 'numpy' library, the third parameter
    in its signature is 'out', which takes either an ndarray or 'None', so
    check if the 'skipna' parameter is either an instance of ndarray or is
    None, since 'skipna' itself should be a boolean
    """
    skipna, args = process_skipna(skipna, args)
    validate_argmin(args, kwargs)
    return skipna