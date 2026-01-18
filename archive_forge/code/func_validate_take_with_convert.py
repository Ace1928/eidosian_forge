from __future__ import annotations
from typing import (
import numpy as np
from numpy import ndarray
from pandas._libs.lib import (
from pandas.errors import UnsupportedFunctionCall
from pandas.util._validators import (
def validate_take_with_convert(convert: ndarray | bool | None, args, kwargs) -> bool:
    """
    If this function is called via the 'numpy' library, the third parameter in
    its signature is 'axis', which takes either an ndarray or 'None', so check
    if the 'convert' parameter is either an instance of ndarray or is None
    """
    if isinstance(convert, ndarray) or convert is None:
        args = (convert,) + args
        convert = True
    validate_take(args, kwargs, max_fname_arg_count=3, method='both')
    return convert