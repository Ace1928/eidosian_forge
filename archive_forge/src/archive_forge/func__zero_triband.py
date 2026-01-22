import numpy as np
import numpy.linalg as L
from scipy.linalg import solveh_banded
from scipy.optimize import golden
from models import _hbspline     #removed because this was segfaulting
import warnings
def _zero_triband(a, lower=0):
    """
    Explicitly zero out unused elements of a real symmetric banded matrix.

    INPUTS:
       a   -- a real symmetric banded matrix (either upper or lower hald)
       lower   -- if True, a is assumed to be the lower half
    """
    nrow, ncol = a.shape
    if lower:
        for i in range(nrow):
            a[i, ncol - i:] = 0.0
    else:
        for i in range(nrow):
            a[i, 0:i] = 0.0
    return a