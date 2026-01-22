from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class NumbaUtilError(Exception):
    """
    Error raised for unsupported Numba engine routines.

    Examples
    --------
    >>> df = pd.DataFrame({"key": ["a", "a", "b", "b"], "data": [1, 2, 3, 4]},
    ...                   columns=["key", "data"])
    >>> def incorrect_function(x):
    ...     return sum(x) * 2.7
    >>> df.groupby("key").agg(incorrect_function, engine="numba")
    Traceback (most recent call last):
    NumbaUtilError: The first 2 arguments to incorrect_function
    must be ['values', 'index']
    """