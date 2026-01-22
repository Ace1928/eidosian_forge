from __future__ import annotations
from collections.abc import (
from typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import (
def _excel2num(x: str) -> int:
    """
    Convert Excel column name like 'AB' to 0-based column index.

    Parameters
    ----------
    x : str
        The Excel column name to convert to a 0-based column index.

    Returns
    -------
    num : int
        The column index corresponding to the name.

    Raises
    ------
    ValueError
        Part of the Excel column name was invalid.
    """
    index = 0
    for c in x.upper().strip():
        cp = ord(c)
        if cp < ord('A') or cp > ord('Z'):
            raise ValueError(f'Invalid column name: {x}')
        index = index * 26 + cp - ord('A') + 1
    return index - 1