from __future__ import annotations
import math as _math  # unexport, see #17
import typing
from functools import partial as _partial
from functools import wraps as _wraps  # unexport, see #17
def _y_to_l(y: float) -> float:
    if y <= _epsilon:
        return y / _ref_y * _kappa
    return 116 * (y / _ref_y) ** (1 / 3) - 16