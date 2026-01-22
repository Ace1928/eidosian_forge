from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class CSSWarning(UserWarning):
    """
    Warning is raised when converting css styling fails.

    This can be due to the styling not having an equivalent value or because the
    styling isn't properly formatted.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1]})
    >>> df.style.applymap(
    ...     lambda x: 'background-color: blueGreenRed;'
    ... ).to_excel('styled.xlsx')  # doctest: +SKIP
    CSSWarning: Unhandled color format: 'blueGreenRed'
    >>> df.style.applymap(
    ...     lambda x: 'border: 1px solid red red;'
    ... ).to_excel('styled.xlsx')  # doctest: +SKIP
    CSSWarning: Unhandled color format: 'blueGreenRed'
    """