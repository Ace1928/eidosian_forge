from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class InvalidIndexError(Exception):
    """
    Exception raised when attempting to use an invalid index key.

    Examples
    --------
    >>> idx = pd.MultiIndex.from_product([["x", "y"], [0, 1]])
    >>> df = pd.DataFrame([[1, 1, 2, 2],
    ...                   [3, 3, 4, 4]], columns=idx)
    >>> df
        x       y
        0   1   0   1
    0   1   1   2   2
    1   3   3   4   4
    >>> df[:, 0]
    Traceback (most recent call last):
    InvalidIndexError: (slice(None, None, None), 0)
    """