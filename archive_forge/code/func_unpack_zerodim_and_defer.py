from __future__ import annotations
from functools import wraps
from typing import (
from pandas._libs.lib import item_from_zerodim
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.generic import (
def unpack_zerodim_and_defer(name: str) -> Callable[[F], F]:
    """
    Boilerplate for pandas conventions in arithmetic and comparison methods.

    Parameters
    ----------
    name : str

    Returns
    -------
    decorator
    """

    def wrapper(method: F) -> F:
        return _unpack_zerodim_and_defer(method, name)
    return wrapper