from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.common import is_list_like
def tile_compat(arr: NumpyIndexT, num: int) -> NumpyIndexT:
    """
    Index compat for np.tile.

    Notes
    -----
    Does not support multi-dimensional `num`.
    """
    if isinstance(arr, np.ndarray):
        return np.tile(arr, num)
    taker = np.tile(np.arange(len(arr)), num)
    return arr.take(taker)