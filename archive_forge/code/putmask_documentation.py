from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import is_list_like
from pandas.core.arrays import ExtensionArray

    Parameters
    ----------
    values : np.ndarray
    num_set : int
        For putmask, this is mask.sum()
    other : Any
    