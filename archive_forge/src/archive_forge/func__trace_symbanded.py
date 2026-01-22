import numpy as np
import numpy.linalg as L
from scipy.linalg import solveh_banded
from scipy.optimize import golden
from models import _hbspline     #removed because this was segfaulting
import warnings
def _trace_symbanded(a, b, lower=0):
    """
    Compute the trace(ab) for two upper or banded real symmetric matrices
    stored either in either upper or lower form.

    INPUTS:
       a, b    -- two banded real symmetric matrices (either lower or upper)
       lower   -- if True, a and b are assumed to be the lower half


    OUTPUTS: trace
       trace   -- trace(ab)
    """
    if lower:
        t = _zero_triband(a * b, lower=1)
        return t[0].sum() + 2 * t[1:].sum()
    else:
        t = _zero_triband(a * b, lower=0)
        return t[-1].sum() + 2 * t[:-1].sum()