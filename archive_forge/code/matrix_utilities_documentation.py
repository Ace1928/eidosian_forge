import numbers
import numpy as np
import scipy.sparse as sp
from cvxpy.interface import numpy_interface as np_intf
Check if a real sparse matrix A satisfies A + A.T == 0.

    Parameters
    ----------
    A : array or sparse matrix
        A square matrix.

    Returns
    -------
    check : bool
        The check result.
    