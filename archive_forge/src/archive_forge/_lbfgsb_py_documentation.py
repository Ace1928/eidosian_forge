import numpy as np
from numpy import array, asarray, float64, zeros
from . import _lbfgsb
from ._optimize import (MemoizeJac, OptimizeResult, _call_callback_maybe_halt,
from ._constraints import old_bound_to_new
from scipy.sparse.linalg import LinearOperator
Return a dense array representation of this operator.

        Returns
        -------
        arr : ndarray, shape=(n, n)
            An array with the same shape and containing
            the same data represented by this `LinearOperator`.

        