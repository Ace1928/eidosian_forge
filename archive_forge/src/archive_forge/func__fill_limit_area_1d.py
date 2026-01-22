from __future__ import annotations
from functools import wraps
from typing import (
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import (
def _fill_limit_area_1d(mask: npt.NDArray[np.bool_], limit_area: Literal['outside', 'inside']) -> None:
    """Prepare 1d mask for ffill/bfill with limit_area.

    Caller is responsible for checking at least one value of mask is False.
    When called, mask will no longer faithfully represent when
    the corresponding are NA or not.

    Parameters
    ----------
    mask : np.ndarray[bool, ndim=1]
        Mask representing NA values when filling.
    limit_area : { "outside", "inside" }
        Whether to limit filling to outside or inside the outer most non-NA value.
    """
    neg_mask = ~mask
    first = neg_mask.argmax()
    last = len(neg_mask) - neg_mask[::-1].argmax() - 1
    if limit_area == 'inside':
        mask[:first] = False
        mask[last + 1:] = False
    elif limit_area == 'outside':
        mask[first + 1:last] = False