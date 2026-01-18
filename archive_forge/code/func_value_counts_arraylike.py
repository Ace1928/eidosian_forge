from __future__ import annotations
import decimal
import operator
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core.array_algos.take import take_nd
from pandas.core.construction import (
from pandas.core.indexers import validate_indices
def value_counts_arraylike(values: np.ndarray, dropna: bool, mask: npt.NDArray[np.bool_] | None=None) -> tuple[ArrayLike, npt.NDArray[np.int64], int]:
    """
    Parameters
    ----------
    values : np.ndarray
    dropna : bool
    mask : np.ndarray[bool] or None, default None

    Returns
    -------
    uniques : np.ndarray
    counts : np.ndarray[np.int64]
    """
    original = values
    values = _ensure_data(values)
    keys, counts, na_counter = htable.value_count(values, dropna, mask=mask)
    if needs_i8_conversion(original.dtype):
        if dropna:
            mask = keys != iNaT
            keys, counts = (keys[mask], counts[mask])
    res_keys = _reconstruct_data(keys, original.dtype, original)
    return (res_keys, counts, na_counter)