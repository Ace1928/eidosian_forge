from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from mizani._colors import SegmentFunctionMap
def _gpf_32(x: NDArrayFloat) -> NDArrayFloat:
    ret = np.zeros(len(x))
    m = x < 0.25
    ret[m] = 4 * x[m]
    m = (x >= 0.25) & (x < 0.92)
    ret[m] = -2 * x[m] + 1.84
    m = x >= 0.92
    ret[m] = x[m] / 0.08 - 11.5
    return ret