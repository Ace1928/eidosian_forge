from __future__ import annotations
from collections.abc import (
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.common import (
def validate_insert_loc(loc: int, length: int) -> int:
    """
    Check that we have an integer between -length and length, inclusive.

    Standardize negative loc to within [0, length].

    The exceptions we raise on failure match np.insert.
    """
    if not is_integer(loc):
        raise TypeError(f'loc must be an integer between -{length} and {length}')
    if loc < 0:
        loc += length
    if not 0 <= loc <= length:
        raise IndexError(f'loc must be an integer between -{length} and {length}')
    return loc