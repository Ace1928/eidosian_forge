from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import is_list_like
from pandas.core.arrays import ExtensionArray
def validate_putmask(values: ArrayLike | MultiIndex, mask: np.ndarray) -> tuple[npt.NDArray[np.bool_], bool]:
    """
    Validate mask and check if this putmask operation is a no-op.
    """
    mask = extract_bool_array(mask)
    if mask.shape != values.shape:
        raise ValueError('putmask: mask and data must be the same size')
    noop = not mask.any()
    return (mask, noop)