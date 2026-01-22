from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class LossySetitemError(Exception):
    """
    Raised when trying to do a __setitem__ on an np.ndarray that is not lossless.

    Notes
    -----
    This is an internal error.
    """