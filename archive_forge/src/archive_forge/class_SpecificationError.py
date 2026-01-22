from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class SpecificationError(Exception):
    """
    Exception raised by ``agg`` when the functions are ill-specified.

    The exception raised in two scenarios.

    The first way is calling ``agg`` on a
    Dataframe or Series using a nested renamer (dict-of-dict).

    The second way is calling ``agg`` on a Dataframe with duplicated functions
    names without assigning column name.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1, 2, 2],
    ...                    'B': range(5),
    ...                    'C': range(5)})
    >>> df.groupby('A').B.agg({'foo': 'count'}) # doctest: +SKIP
    ... # SpecificationError: nested renamer is not supported

    >>> df.groupby('A').agg({'B': {'foo': ['sum', 'max']}}) # doctest: +SKIP
    ... # SpecificationError: nested renamer is not supported

    >>> df.groupby('A').agg(['min', 'min']) # doctest: +SKIP
    ... # SpecificationError: nested renamer is not supported
    """