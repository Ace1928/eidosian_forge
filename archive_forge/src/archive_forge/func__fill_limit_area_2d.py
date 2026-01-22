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
def _fill_limit_area_2d(mask: npt.NDArray[np.bool_], limit_area: Literal['outside', 'inside']) -> None:
    """Prepare 2d mask for ffill/bfill with limit_area.

    When called, mask will no longer faithfully represent when
    the corresponding are NA or not.

    Parameters
    ----------
    mask : np.ndarray[bool, ndim=1]
        Mask representing NA values when filling.
    limit_area : { "outside", "inside" }
        Whether to limit filling to outside or inside the outer most non-NA value.
    """
    neg_mask = ~mask.T
    if limit_area == 'outside':
        la_mask = np.maximum.accumulate(neg_mask, axis=0) & np.maximum.accumulate(neg_mask[::-1], axis=0)[::-1]
    else:
        la_mask = ~np.maximum.accumulate(neg_mask, axis=0) | ~np.maximum.accumulate(neg_mask[::-1], axis=0)[::-1]
    mask[la_mask.T] = False