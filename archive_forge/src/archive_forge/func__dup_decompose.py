from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def _dup_decompose(f, K):
    """Helper function for :func:`dup_decompose`."""
    df = len(f) - 1
    for s in range(2, df):
        if df % s != 0:
            continue
        h = _dup_right_decompose(f, s, K)
        if h is not None:
            g = _dup_left_decompose(f, h, K)
            if g is not None:
                return (g, h)
    return None