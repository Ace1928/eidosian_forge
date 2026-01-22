import numpy as np
import numpy.linalg as L
from scipy.linalg import solveh_banded
from scipy.optimize import golden
from models import _hbspline     #removed because this was segfaulting
import warnings
def _triangle2unit(tb, lower=0):
    """
    Take a banded triangular matrix and return its diagonal and the
    unit matrix: the banded triangular matrix with 1's on the diagonal,
    i.e. each row is divided by the corresponding entry on the diagonal.

    INPUTS:
       tb    -- a lower triangular banded matrix
       lower -- if True, then tb is assumed to be lower triangular banded,
                in which case return value is also lower triangular banded.

    OUTPUTS: d, b
       d     -- diagonal entries of tb
       b     -- unit matrix: if lower is False, b is upper triangular
                banded and its rows of have been divided by d,
                else lower is True, b is lower triangular banded
                and its columns have been divieed by d.
    """
    if lower:
        d = tb[0].copy()
    else:
        d = tb[-1].copy()
    if lower:
        return (d, tb / d)
    else:
        lnum = _upper2lower(tb)
        return (d, _lower2upper(lnum / d))