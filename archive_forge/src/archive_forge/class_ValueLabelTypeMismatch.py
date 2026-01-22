from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class ValueLabelTypeMismatch(Warning):
    """
    Warning raised by to_stata on a category column that contains non-string values.

    Examples
    --------
    >>> df = pd.DataFrame({"categories": pd.Series(["a", 2], dtype="category")})
    >>> df.to_stata('test') # doctest: +SKIP
    ... # ValueLabelTypeMismatch: Stata value labels (pandas categories) must be str...
    """