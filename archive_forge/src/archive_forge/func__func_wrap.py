from __future__ import annotations
from typing import (  # noqa: UP035
import numpy as np
from scipy.optimize import OptimizeResult
from ._constraints import old_bound_to_new, Bounds
from ._direct import direct as _direct  # type: ignore
def _func_wrap(x, args=None):
    x = np.asarray(x)
    if args is None:
        f = func(x)
    else:
        f = func(x, *args)
    return np.asarray(f).item()