from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas._typing import (
from pandas.util._validators import validate_percentile
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.reshape.concat import concat
from pandas.io.formats.format import format_percentiles
def _refine_percentiles(percentiles: Sequence[float] | np.ndarray | None) -> npt.NDArray[np.float64]:
    """
    Ensure that percentiles are unique and sorted.

    Parameters
    ----------
    percentiles : list-like of numbers, optional
        The percentiles to include in the output.
    """
    if percentiles is None:
        return np.array([0.25, 0.5, 0.75])
    percentiles = list(percentiles)
    validate_percentile(percentiles)
    if 0.5 not in percentiles:
        percentiles.append(0.5)
    percentiles = np.asarray(percentiles)
    unique_pcts = np.unique(percentiles)
    assert percentiles is not None
    if len(unique_pcts) < len(percentiles):
        raise ValueError('percentiles cannot contain duplicates')
    return unique_pcts