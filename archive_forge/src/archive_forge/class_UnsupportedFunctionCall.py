from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class UnsupportedFunctionCall(ValueError):
    """
    Exception raised when attempting to call a unsupported numpy function.

    For example, ``np.cumsum(groupby_object)``.

    Examples
    --------
    >>> df = pd.DataFrame({"A": [0, 0, 1, 1],
    ...                    "B": ["x", "x", "z", "y"],
    ...                    "C": [1, 2, 3, 4]}
    ...                   )
    >>> np.cumsum(df.groupby(["A"]))
    Traceback (most recent call last):
    UnsupportedFunctionCall: numpy operations are not valid with groupby.
    Use .groupby(...).cumsum() instead
    """