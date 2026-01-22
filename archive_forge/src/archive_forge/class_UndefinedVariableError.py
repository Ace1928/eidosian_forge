from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class UndefinedVariableError(NameError):
    """
    Exception raised by ``query`` or ``eval`` when using an undefined variable name.

    It will also specify whether the undefined variable is local or not.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1]})
    >>> df.query("A > x") # doctest: +SKIP
    ... # UndefinedVariableError: name 'x' is not defined
    >>> df.query("A > @y") # doctest: +SKIP
    ... # UndefinedVariableError: local variable 'y' is not defined
    >>> pd.eval('x + 1') # doctest: +SKIP
    ... # UndefinedVariableError: name 'x' is not defined
    """

    def __init__(self, name: str, is_local: bool | None=None) -> None:
        base_msg = f'{repr(name)} is not defined'
        if is_local:
            msg = f'local variable {base_msg}'
        else:
            msg = f'name {base_msg}'
        super().__init__(msg)