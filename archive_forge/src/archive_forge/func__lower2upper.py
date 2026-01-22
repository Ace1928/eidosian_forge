import numpy as np
import numpy.linalg as L
from scipy.linalg import solveh_banded
from scipy.optimize import golden
from models import _hbspline     #removed because this was segfaulting
import warnings
def _lower2upper(lb):
    """
    Convert lower triangular banded matrix to upper banded form.

    INPUTS:
       lb  -- a lower triangular banded matrix

    OUTPUTS: ub
       ub  -- an upper triangular banded matrix with same entries
              as lb
    """
    ub = np.zeros(lb.shape, lb.dtype)
    nrow, ncol = lb.shape
    for i in range(lb.shape[0]):
        ub[nrow - 1 - i, i:ncol] = lb[i, 0:ncol - i]
        ub[nrow - 1 - i, 0:i] = lb[i, ncol - i:]
    return ub